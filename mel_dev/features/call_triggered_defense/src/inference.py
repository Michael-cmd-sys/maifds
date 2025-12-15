from __future__ import annotations

import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from typing import Any, Dict, Optional

from .config import FEATURE_COLUMNS
from .model import CallTriggeredDefenseModel
from .rules import CallTxEvent, high_precision_rule

# Optional: set context once (safe for CPU)
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

# -------------------------------------------------------------------
# Simple cache so we don't reload the checkpoint every request
# -------------------------------------------------------------------
_MODEL_CACHE: Optional[CallTriggeredDefenseModel] = None
_MODEL_PATH_CACHE: Optional[str] = None


def load_model(model_path: str | None = None) -> CallTriggeredDefenseModel:
    """
    Load the trained MindSpore model for inference.
    Uses an absolute path relative to this file so it works under uvicorn.
    Cached after first load.
    """
    global _MODEL_CACHE, _MODEL_PATH_CACHE

    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "call_triggered_defense_mlp.ckpt")

    # return cached if same path
    if _MODEL_CACHE is not None and _MODEL_PATH_CACHE == model_path:
        return _MODEL_CACHE

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    input_dim = len(FEATURE_COLUMNS)
    model = CallTriggeredDefenseModel(input_dim=input_dim)

    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)

    _MODEL_CACHE = model
    _MODEL_PATH_CACHE = model_path
    return model


def prepare_features(event_dict: Dict[str, Any]) -> np.ndarray:
    """
    Convert event dictionary into a model-ready feature vector.
    Missing fields default to 0.
    """
    x = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLUMNS):
        val = event_dict.get(col, 0.0)
        try:
            x[0, i] = float(val)
        except Exception:
            x[0, i] = 0.0
    return x


def _sigmoid(z: float) -> float:
    # numerically stable sigmoid
    if z >= 0:
        ez = np.exp(-z)
        return float(1.0 / (1.0 + ez))
    ez = np.exp(z)
    return float(ez / (1.0 + ez))


def predict_proba(model: CallTriggeredDefenseModel, x_np: np.ndarray) -> Dict[str, float]:
    """
    Run model and return probability.
    If model output looks like logits, sigmoid is applied.
    """
    x_tensor = Tensor(x_np, ms.float32)
    raw = float(model(x_tensor).asnumpy().flatten()[0])

    # If output already in [0, 1], treat as probability; else sigmoid it.
    if 0.0 <= raw <= 1.0:
        prob = raw
    else:
        prob = _sigmoid(raw)

    return {"prob": float(prob), "raw": float(raw)}


def enrich_event(event_dict: dict) -> dict:
    # Ensure amount exists
    if "amount" not in event_dict and "tx_amount" in event_dict:
        event_dict["amount"] = event_dict["tx_amount"]

    amount = float(event_dict.get("amount", 0.0))
    call_delta = float(event_dict.get("call_to_tx_delta_seconds", 999999.0))

    # Simple derived features (tune thresholds later)
    event_dict.setdefault("has_recent_call", 1.0 if call_delta <= 300 else 0.0)
    event_dict.setdefault("is_large_amount", 1.0 if amount >= 1000 else 0.0)

    # Defaults for fields your model expects
    event_dict.setdefault("transaction_day", 0.0)
    event_dict.setdefault("transaction_hour", 0.0)
    event_dict.setdefault("call_duration_seconds", 0.0)
    event_dict.setdefault("device_age_days", 0.0)

    # balances (only if you add them later)
    event_dict.setdefault("oldbalanceOrg", 0.0)
    event_dict.setdefault("newbalanceOrig", 0.0)
    event_dict.setdefault("oldbalanceDest", 0.0)
    event_dict.setdefault("newbalanceDest", 0.0)
    event_dict.setdefault("origin_balance_delta", 0.0)
    event_dict.setdefault("dest_balance_delta", 0.0)

    return event_dict


def run_inference(event_dict: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    """
    Full inference: rules + model ensemble.
    """
    # Accept both "tx_amount" and "amount"
    if "amount" not in event_dict and "tx_amount" in event_dict:
        event_dict["amount"] = event_dict["tx_amount"]

    # ✅ Enrich missing features for the model
    event_dict = enrich_event(event_dict)

    # Ensure numeric defaults (avoid None reaching rules)
    call_delta = float(event_dict.get("call_to_tx_delta_seconds", 999999.0))
    nlp_score = float(event_dict.get("nlp_suspicion_score", 0.0))

    # 1) Prepare features
    x = prepare_features(event_dict)

    # 2) Load model (cached)
    model = load_model()

    # 3) Model score
    pred = predict_proba(model, x)
    model_score = pred["prob"]

    # 4) Rule engine input
    rule_event = CallTxEvent(
        call_to_tx_delta_seconds=call_delta,
        recipient_first_time=int(event_dict.get("recipient_first_time", 0)),
        tx_amount=float(event_dict.get("amount", 0.0)),
        contact_list_flag=int(event_dict.get("contact_list_flag", 0)),
        nlp_suspicion_score=nlp_score,
    )

    rule_flag = bool(high_precision_rule(rule_event))

    # 5) Ensemble logic
    if rule_flag:
        risk_level = "HIGH"
        reason = "Rule-based high-risk pattern detected"
    else:
        if model_score >= 0.8:
            risk_level = "HIGH"
            reason = "Model indicates high fraud probability"
        elif model_score >= 0.4:
            risk_level = "MEDIUM"
            reason = "Model indicates moderate fraud risk"
        else:
            risk_level = "LOW"
            reason = "Model indicates low fraud probability"

    actions_map = {
        "HIGH": ["SMS_USER_ALERT", "TEMP_HOLD", "NOTIFY_TELCO"],
        "MEDIUM": ["SMS_USER_ALERT"],
        "LOW": [],
    }

    resp: Dict[str, Any] = {
        "fraud_probability": float(model_score),
        "rule_flag": rule_flag,
        "risk_level": risk_level,
        "reason": reason,
        "actions": actions_map[risk_level],
    }

    if debug:
        resp["debug"] = {
            "feature_columns": FEATURE_COLUMNS,
            "model_input_vector": x.flatten().tolist(),
            "raw_model_output": pred["raw"],
            "checkpoint_path": _MODEL_PATH_CACHE,
        }

    return resp
