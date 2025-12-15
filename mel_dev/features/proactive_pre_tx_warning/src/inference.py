from __future__ import annotations

import os
import numpy as np
import mindspore as ms
from mindspore import Tensor
from typing import Any, Dict, Tuple

from .config import FEATURE_COLUMNS, MODEL_CKPT_PATH, MEAN_PATH, STD_PATH
from .model import ProactiveWarningModel
from .rules import UserRiskProfile, must_warn_user, build_warning_message

# set once (CPU)
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

_CACHE = {
    "model": None,
    "mean": None,
    "std": None,
    "ckpt_path": None,
}


def load_training_stats() -> Tuple[np.ndarray, np.ndarray]:
    if _CACHE["mean"] is not None:
        return _CACHE["mean"], _CACHE["std"]

    if not os.path.exists(MEAN_PATH) or not os.path.exists(STD_PATH):
        # fallback: no normalization (still works)
        mean = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
        std = np.ones((1, len(FEATURE_COLUMNS)), dtype=np.float32)
        _CACHE["mean"], _CACHE["std"] = mean, std
        return mean, std

    mean = np.load(MEAN_PATH).astype(np.float32)
    std = np.load(STD_PATH).astype(np.float32)

    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std[std == 0] = 1.0
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)

    _CACHE["mean"], _CACHE["std"] = mean, std
    return mean, std


def load_model() -> ProactiveWarningModel:
    if _CACHE["model"] is not None:
        return _CACHE["model"]

    input_dim = len(FEATURE_COLUMNS)
    model = ProactiveWarningModel(input_dim=input_dim)
    model.set_train(False)

    ckpt_path = MODEL_CKPT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(model, param_dict)

    _CACHE["model"] = model
    _CACHE["ckpt_path"] = ckpt_path
    return model


def prepare_features(user_features: Dict[str, Any], mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = np.zeros((1, len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, name in enumerate(FEATURE_COLUMNS):
        try:
            x[0, i] = float(user_features.get(name, 0.0))
        except Exception:
            x[0, i] = 0.0

    x_norm = (x - mean) / std
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = np.clip(x_norm, -10.0, 10.0)
    return x_norm


def predict_warn_probability(model: ProactiveWarningModel, x_norm: np.ndarray) -> float:
    out = model(Tensor(x_norm, ms.float32))
    return float(out.asnumpy().reshape(-1)[0])


def build_profile_from_dict(user_features: Dict[str, Any]) -> UserRiskProfile:
    def get(name, default=0.0):
        return user_features.get(name, default)

    return UserRiskProfile(
        recent_risky_clicks_7d=int(get("recent_risky_clicks_7d", 0)),
        recent_scam_calls_7d=int(get("recent_scam_calls_7d", 0)),
        scam_campaign_intensity=float(get("scam_campaign_intensity", 0.0)),
        device_age_days=float(get("device_age_days", 180.0)),
        is_new_device=int(get("is_new_device", 0)),
        tx_count_total=int(get("tx_count_total", 0)),
        avg_tx_amount=float(get("avg_tx_amount", 0.0)),
        max_tx_amount=float(get("max_tx_amount", 0.0)),
        historical_fraud_flag=int(get("historical_fraud_flag", 0)),
        is_in_campaign_cohort=int(get("is_in_campaign_cohort", 0)),
        user_risk_score=float(get("user_risk_score", 0.0)),
    )


def run_inference(user_features: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    mean, std = load_training_stats()
    model = load_model()

    # 1) ML probability
    x_norm = prepare_features(user_features, mean, std)
    prob = predict_warn_probability(model, x_norm)

    # 2) Rule-based decision
    profile = build_profile_from_dict(user_features)
    rule_flag = bool(must_warn_user(profile))

    # 3) Ensemble: rule overrides model
    if rule_flag:
        should_warn = True
        decision_source = "RULE_OVERRIDE"
        risk_tier = "HIGH"
    else:
        if prob >= 0.8:
            should_warn = True
            decision_source = "MODEL_HIGH_CONFIDENCE"
            risk_tier = "HIGH"
        elif prob >= 0.4:
            should_warn = True
            decision_source = "MODEL_MEDIUM_CONFIDENCE"
            risk_tier = "MEDIUM"
        else:
            should_warn = False
            decision_source = "MODEL_LOW_RISK"
            risk_tier = "LOW"

    sms_text = build_warning_message(profile) if should_warn else ""

    actions = []
    if should_warn:
        actions = ["SEND_PROACTIVE_SMS"]
        if risk_tier == "HIGH":
            actions.append("ENABLE_STRICT_VERIFICATION_WINDOW")

    resp = {
        "should_warn_probability": float(prob),
        "rule_flag": rule_flag,
        "should_warn": should_warn,
        "risk_tier": risk_tier,
        "decision_source": decision_source,
        "actions": actions,
        "sms_text": sms_text,
    }

    if debug:
        resp["debug"] = {
            "feature_columns": FEATURE_COLUMNS,
            "normalized_vector": x_norm.reshape(-1).tolist(),
            "ckpt_path": _CACHE["ckpt_path"],
            "event_used": user_features,
        }

    return resp
