import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor

from config import (
    TRAINING_TABLE_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
)
from model import ClickTxLinkModel
from rules import ClickTxEvent, high_precision_click_rule


# -----------------------------
# Helpers: load data & model
# -----------------------------

def load_training_stats():
    """
    Load processed training table to compute feature normalization stats
    and median transaction amount for rule logic.
    """
    df = pd.read_parquet(TRAINING_TABLE_PATH)

    X = df[FEATURE_COLUMNS].astype("float32").values
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    median_amount = float(df["amount"].median())

    return mean, std, median_amount


def load_model():
    """
    Load ClickTxLinkModel and restore checkpoint.
    """
    mean, std, median_amount = load_training_stats()
    input_dim = len(FEATURE_COLUMNS)

    model = ClickTxLinkModel(input_dim=input_dim)
    ckpt_name = "click_tx_link_model.ckpt"
    ms.load_checkpoint(ckpt_name, model)

    return model, mean, std, median_amount


# -----------------------------
# Core prediction helpers
# -----------------------------

def prepare_features(event: dict, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Build normalized feature vector (1, D) from event dict.
    Missing fields are filled with zeros.
    """
    x = np.zeros((1, len(FEATURE_COLUMNS)), dtype="float32")

    for i, name in enumerate(FEATURE_COLUMNS):
        if name in event:
            x[0, i] = float(event[name])
        # else: leave as 0

    # normalize
    x_norm = (x - mean) / std
    return x_norm


def predict_proba(model, x_norm: np.ndarray) -> float:
    """
    Run model forward pass and return fraud probability.
    """
    model.set_train(False)
    logits = model(Tensor(x_norm, ms.float32))
    prob = float(logits.asnumpy()[0, 0])
    return prob


# -----------------------------
# Public inference entry point
# -----------------------------

def run_inference(event: dict) -> dict:
    """
    Full inference pipeline:
      - prepare features
      - load model + stats
      - get ML fraud probability
      - apply high-precision click→tx rule
      - combine into risk_level + recommended actions
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    model, mean, std, median_amount = load_model()

    # 1) Model score
    x_norm = prepare_features(event, mean, std)
    fraud_prob = predict_proba(model, x_norm)

    # 2) Rule evaluation
    rule_event = ClickTxEvent(
        time_between_click_and_tx=float(event.get("time_between_click_and_tx", 99999.0)),
        url_risk_score=float(event.get("url_risk_score", 0.0)),
        url_reported_flag=int(event.get("url_reported_flag", 0)),
        amount=float(event.get("amount", 0.0)),
        median_amount=median_amount,
        clicked_recently=int(event.get("clicked_recently", 0)),
        device_click_count_1d=int(event.get("device_click_count_1d", 0)),
        user_click_count_1d=int(event.get("user_click_count_1d", 0)),
    )
    rule_flag = high_precision_click_rule(rule_event)

    # 3) Ensemble: rules + model
    if rule_flag:
        risk_level = "HIGH"
        reason = "Recent click on high-risk / reported URL with large transaction."
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

    # 4) Actions mapping
    if risk_level == "HIGH":
        actions = [
            "HIGH_RISK_HOLD",
            "SMS_USER_WARNING",
            "NOTIFY_TELCO_OPS",
        ]
    elif risk_level == "MEDIUM":
        actions = [
            "SMS_USER_WARNING",
        ]
    else:
        actions = []

    return {
        "fraud_probability": fraud_prob,
        "rule_flag": rule_flag,
        "risk_level": risk_level,
        "reason": reason,
        "actions": actions,
    }
