from __future__ import annotations

import os
import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor
from typing import Any, Dict, Optional, Tuple

from .config import TRAINING_TABLE_PATH, FEATURE_COLUMNS, MODEL_CKPT_PATH
from .model import ClickTxLinkModel
from .rules import ClickTxEvent, high_precision_click_rule

# set once (CPU)
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

# -------------------------------------------------------------------
# Cache to avoid reloading parquet/checkpoint for every request
# -------------------------------------------------------------------
_CACHE = {
    "model": None,
    "mean": None,
    "std": None,
    "median_amount": None,
    "ckpt_path": None,
    "stats_mtime": None,
}


def enrich_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required fields exist using safe defaults.
    Also map tx_amount -> amount (gateway may send tx_amount).
    """
    if "amount" not in event and "tx_amount" in event:
        event["amount"] = event["tx_amount"]

    # Defaults for expected model/rule keys
    event.setdefault("time_since_last_tx_seconds", 0.0)
    event.setdefault("tx_hour", 0.0)
    event.setdefault("tx_dayofweek", 0.0)

    event.setdefault("url_hash_numeric", 0.0)
    event.setdefault("url_risk_score", 0.0)
    event.setdefault("url_reported_flag", 0)
    event.setdefault("clicked_recently", 0)
    event.setdefault("time_between_click_and_tx", 999999.0)

    event.setdefault("device_click_count_1d", 0)
    event.setdefault("user_click_count_1d", 0)

    return event


def _apply_feature_transforms(df_or_x):
    """
    Apply the same feature transforms used at inference time.
    For now: log1p(url_hash_numeric).
    Works for both pandas DF and numpy array.
    """
    if "url_hash_numeric" in FEATURE_COLUMNS:
        idx = FEATURE_COLUMNS.index("url_hash_numeric")
        if hasattr(df_or_x, "iloc"):  # pandas
            s = df_or_x["url_hash_numeric"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df_or_x["url_hash_numeric"] = np.log1p(np.clip(s.to_numpy(dtype=np.float64), 0, None))
        else:  # numpy
            df_or_x[:, idx] = np.log1p(np.clip(df_or_x[:, idx], 0, None))
    return df_or_x



def load_training_stats() -> Tuple[np.ndarray, np.ndarray, float]:
    if not os.path.exists(TRAINING_TABLE_PATH):
        mean = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
        std = np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32)
        median_amount = 253.11  # better default than 1.0
        _CACHE["mean"], _CACHE["std"], _CACHE["median_amount"] = mean, std, median_amount
        _CACHE["stats_mtime"] = None
        return mean, std, median_amount

    mtime = os.path.getmtime(TRAINING_TABLE_PATH)

    # ✅ use cached only if parquet hasn't changed
    if _CACHE["mean"] is not None and _CACHE["stats_mtime"] == mtime:
        return _CACHE["mean"], _CACHE["std"], _CACHE["median_amount"]

    df = pd.read_parquet(TRAINING_TABLE_PATH).copy()

    # sanitize
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ✅ apply same transforms as inference BEFORE stats
    df = _apply_feature_transforms(df)

    # ✅ float64 for stable stats (avoid overflow), then cast to float32
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)

    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std == 0] = 1.0

    # median for rules (positive amounts only)
    amt = df["amount"].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    amt_pos = amt[amt > 0]
    median_amount = float(np.median(amt_pos)) if amt_pos.size > 0 else 253.11

    if not np.isfinite(median_amount) or median_amount <= 0:
        median_amount = 253.11

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    _CACHE["mean"], _CACHE["std"], _CACHE["median_amount"] = mean, std, median_amount
    _CACHE["stats_mtime"] = mtime
    return mean, std, median_amount




def load_model() -> ClickTxLinkModel:
    """
    Load model checkpoint (cached).
    """
    if _CACHE["model"] is not None:
        return _CACHE["model"]

    input_dim = len(FEATURE_COLUMNS)
    model = ClickTxLinkModel(input_dim=input_dim)
    model.set_train(False)

    ckpt_path = MODEL_CKPT_PATH
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(__file__), ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)

    _CACHE["model"] = model
    _CACHE["ckpt_path"] = ckpt_path
    return model


def prepare_features(event: Dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)

    # fill raw features
    for i, name in enumerate(FEATURE_COLUMNS):
        v = event.get(name, 0.0)
        try:
            x[0, i] = float(v)
        except Exception:
            x[0, i] = 0.0

    # ✅ hash/id transform (prevents huge values)
    if "url_hash_numeric" in FEATURE_COLUMNS:
        idx = FEATURE_COLUMNS.index("url_hash_numeric")
        v = float(x[0, idx])
        if not np.isfinite(v) or v < 0:
            v = 0.0
        x[0, idx] = np.log1p(v)

    # ✅ log-transform time features (must match training transforms)
    for col in ["time_between_click_and_tx", "time_since_last_tx_seconds"]:
        if col in FEATURE_COLUMNS:
            idx = FEATURE_COLUMNS.index(col)
            v = float(x[0, idx])
            if not np.isfinite(v) or v < 0:
                v = 0.0
            x[0, idx] = np.log1p(v)

    # normalize
    x_norm = (x - mean) / std
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

    # clip outliers
    x_norm = np.clip(x_norm, -10.0, 10.0)

    return x_norm


def predict_proba(model: ClickTxLinkModel, x_norm: np.ndarray) -> float:
    model.set_train(False)
    logits = float(model(Tensor(x_norm, ms.float32)).asnumpy().reshape(-1)[0])

    # Convert logits -> probability
    prob = float(1.0 / (1.0 + np.exp(-logits)))
    return prob




def run_inference(event: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """
    Full inference:
      - enrich event
      - normalize features
      - ML score
      - rule override
    """
    event = enrich_event(event)

    mean, std, median_amount = load_training_stats()
    model = load_model()

    x_norm = prepare_features(event, mean, std)
    fraud_prob = predict_proba(model, x_norm)

    rule_event = ClickTxEvent(
        time_between_click_and_tx=float(event.get("time_between_click_and_tx", 999999.0)),
        url_risk_score=float(event.get("url_risk_score", 0.0)),
        url_reported_flag=int(event.get("url_reported_flag", 0)),
        amount=float(event.get("amount", 0.0)),
        median_amount=float(median_amount),
        clicked_recently=int(event.get("clicked_recently", 0)),
        device_click_count_1d=int(event.get("device_click_count_1d", 0)),
        user_click_count_1d=int(event.get("user_click_count_1d", 0)),
    )
    rule_flag = bool(high_precision_click_rule(rule_event))

    if rule_flag:
        risk_level = "HIGH"
        reason = "Recent click on high-risk/reported URL with large transaction."
    else:
        if fraud_prob >= 0.8:
            risk_level = "HIGH"
            reason = "Model indicates high fraud probability from click→tx pattern."
        elif fraud_prob >= 0.4:
            risk_level = "MEDIUM"
            reason = "Model indicates moderate risk from click→tx pattern."
        else:
            risk_level = "LOW"
            reason = "Model indicates low fraud probability."

    if risk_level == "HIGH":
        actions = ["HIGH_RISK_HOLD", "SMS_USER_WARNING", "NOTIFY_TELCO_OPS"]
    elif risk_level == "MEDIUM":
        actions = ["SMS_USER_WARNING"]
    else:
        actions = []

    resp: Dict[str, Any] = {
        "fraud_probability": float(fraud_prob),
        "rule_flag": rule_flag,
        "risk_level": risk_level,
        "reason": reason,
        "actions": actions,
    }

    if debug:
        resp["debug"] = {
            "feature_columns": FEATURE_COLUMNS,
            "normalized_vector": x_norm.reshape(-1).tolist(),
            "ckpt_path": _CACHE["ckpt_path"],
            "median_amount": float(median_amount),
            "event_used": event,
        }

    return resp
