import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor

from config import (
    TRAINING_TABLE_PATH,
    FEATURE_COLUMNS,
)
from model import ProactiveWarningModel
from rules import UserRiskProfile, must_warn_user, build_warning_message


def load_training_stats():
    """
    Load feature normalization stats and a sample df (for safety if needed).
    """
    # for now, we trust feature_mean.npy & feature_std.npy exist in src/
    mean = np.load("feature_mean.npy")
    std = np.load("feature_std.npy")
    std[std == 0] = 1.0

    return mean, std


def load_model():
    """
    Instantiate model and load checkpoint.
    """
    mean, std = load_training_stats()
    input_dim = len(FEATURE_COLUMNS)

    model = ProactiveWarningModel(input_dim=input_dim)
    ckpt_name = "proactive_warning_mlp.ckpt"
    ms.load_checkpoint(ckpt_name, model)

    return model, mean, std


def prepare_features(user_features: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Build normalized feature vector (1, D) from user_features dict.
    """
    x = np.zeros((1, len(FEATURE_COLUMNS)), dtype="float32")
    for i, name in enumerate(FEATURE_COLUMNS):
        if name in user_features:
            x[0, i] = float(user_features[name])
    x_norm = (x - mean) / std
    return x_norm


def predict_warn_probability(model, x_norm: np.ndarray) -> float:
    """
    Run model forward and return probability of 'should_warn'.
    """
    model.set_train(False)
    logits = model(Tensor(x_norm, ms.float32))
    prob = float(logits.asnumpy()[0, 0])
    return prob


def build_profile_from_dict(user_features: dict) -> UserRiskProfile:
    """
    Convert raw dict into UserRiskProfile for rule evaluation.
    Missing fields are defaulted sensibly.
    """
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


def run_inference(user_features: dict) -> dict:
    """
    Full proactive warning inference:
      - load model + stats
      - compute should_warn_probability
      - evaluate high-precision rule
      - decide final 'should_warn' flag + SMS content + action tier
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    model, mean, std = load_model()

    # 1) ML probability
    x_norm = prepare_features(user_features, mean, std)
    prob = predict_warn_probability(model, x_norm)

    # 2) Rule-based decision
    profile = build_profile_from_dict(user_features)
    rule_flag = must_warn_user(profile)

    # 3) Ensemble: rule overrides model
    if rule_flag:
        should_warn = True
        decision_source = "RULE_AND_MODEL"
        risk_tier = "HIGH"
    else:
        # thresholds for proactive messaging
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

    # Actions
    if should_warn:
        actions = ["SEND_PROACTIVE_SMS"]
        if risk_tier == "HIGH":
            actions.append("ENABLE_STRICT_VERIFICATION_WINDOW")
    else:
        actions = []

    return {
        "should_warn_probability": prob,
        "rule_flag": rule_flag,
        "should_warn": should_warn,
        "risk_tier": risk_tier,
        "decision_source": decision_source,
        "actions": actions,
        "sms_text": sms_text,
    }
