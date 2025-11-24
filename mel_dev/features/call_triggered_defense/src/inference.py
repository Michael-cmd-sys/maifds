import numpy as np
import mindspore as ms
from mindspore import Tensor

from config import FEATURE_COLUMNS
from model import CallTriggeredDefenseModel
from rules import CallTxEvent, high_precision_rule


def load_model(model_path="call_triggered_defense_mlp.ckpt"):
    """
    Load the trained MindSpore model for inference.
    """
    input_dim = len(FEATURE_COLUMNS)
    model = CallTriggeredDefenseModel(input_dim=input_dim)
    param_dict = ms.load_checkpoint(model_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    return model


def prepare_features(event_dict):
    """
    Convert a JSON-like event dictionary into a model-ready feature vector.
    Missing fields default to 0.
    """
    x = []
    for col in FEATURE_COLUMNS:
        x.append(float(event_dict.get(col, 0.0)))
    return np.array(x, dtype=np.float32).reshape(1, -1)


def predict_proba(model, x_np):
    """
    Run the MLP model to get fraud probability.
    """
    x_tensor = Tensor(x_np, ms.float32)
    proba = model(x_tensor).asnumpy().flatten()[0]
    return float(proba)


def run_inference(event_dict):
    """
    Full inference: rules + model ensemble.
    """
    # 1. Prepare features for the model
    x = prepare_features(event_dict)

    # 2. Load model
    model = load_model()

    # 3. Model score
    model_score = predict_proba(model, x)

    # 4. Rule engine input
    rule_event = CallTxEvent(
        call_to_tx_delta_seconds=event_dict.get("call_to_tx_delta_seconds", None),
        recipient_first_time=event_dict.get("recipient_first_time", 0),
        tx_amount=event_dict.get("amount", 0.0),
        contact_list_flag=event_dict.get("contact_list_flag", 0),
        nlp_suspicion_score=event_dict.get("nlp_suspicion_score", None),
    )

    rule_flag = high_precision_rule(rule_event)

    # 5. Ensemble logic
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

    # 6. Final response payload
    return {
        "fraud_probability": model_score,
        "rule_flag": rule_flag,
        "risk_level": risk_level,
        "reason": reason,
        "actions": {
            "HIGH": ["SMS_USER_ALERT", "TEMP_HOLD", "NOTIFY_TELCO"],
            "MEDIUM": ["SMS_USER_ALERT"],
            "LOW": []
        }[risk_level],
    }
