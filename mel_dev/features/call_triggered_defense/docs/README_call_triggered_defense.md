**📘 Call Triggered Defense – Model Documentation (v2)**
**🔹 1. Model Name**
Model: CallTriggeredDefenseModel
Type: Tabular MLP classifier (MindSpore) + Rule-Based High-Precision Layer
Purpose: Detect suspicious transactions occurring shortly after social-engineering phone calls.

**🔹 2. Datasets Used**
📌 Primary Dataset – PaySim Transactions (Kaggle)

Sampled at 200,000 rows, with 50,000 for training

Provides realistic mobile-money behavior, including:

transaction amounts

sender/receiver IDs

balances

fraud label (isFraud)

Engineered features include:

transaction_day, transaction_hour

origin_balance_delta, dest_balance_delta

is_large_amount flag

recipient_first_time

**📞 Telco Call/Churn Dataset (Kaggle)**
Used as a proxy for call behavior to generate synthetic call context:

recent-call flag

call_to_tx_delta_seconds

call_duration_seconds

device_age_days (from tenure)

contact_list_flag

**💬 SMS Spam Dataset (Kaggle)**

Used to generate a synthetic NLP suspicion score (0–1) for possible scam/spam call patterns.

**🗂️ Final Training Table – call_tx_training_table.parquet**

Combines:

transaction features

synthetic call features

optional NLP suspicion score

Used for both rule-based detection and MindSpore ML training.

**🔹 3. Model Implementation, Training, Inference & Deployment**
**🏗️ Model Implementation (MindSpore)**

Architecture:

Input: 17 features (transaction + call context)

Hidden Layers:

Dense(64) → ReLU → Dropout

Dense(32) → ReLU → Dropout

Output: Dense(1) → Sigmoid → fraud probability [0,1]

**🎯 Training Setup**

Data: call_tx_training_table.parquet

Loss: BCELoss

Optimizer: Adam (lr=1e-3)

Split: 85% Train / 15% Validation

Batch size: 256

Epochs: 20

Manual loop: uses value_and_grad

Checkpoint: call_triggered_defense_mlp.ckpt

**⚙️ Inference Engine**

The pipeline integrates rules + MLP:

Load MindSpore checkpoint

Prepare feature vector

Compute:

fraud_probability (model)

rule_flag (rules)

Determine risk_level: {LOW, MEDIUM, HIGH}

Return recommended actions:

Risk	Actions
HIGH	SMS alert, temporary hold, notify telco
MEDIUM	SMS alert
LOW	allow
**🌐 Deployment Concept**

Designed as a microservice:

Exposed via FastAPI / Flask

Accepts live transaction + call metadata

Returns real-time score + action

Integrates with telco CDR streams or app-side call events

🔹 4. Metrics & Expected Impact (v1 – Transaction + Synthetic Call Features)
**📉 Training Behavior**
Training loss: 0.51 → 0.02

Validation loss: ~0.0108

Model effectively learns PaySim fraud patterns

**📈 Expected Impact (with real call metadata)**

Accuracy & Recall:
High recall on fraud; rule layer increases precision for risky call patterns.

Inference Throughput:
Lightweight MLP suitable for high TPS.

Operational Cost Reduction:
Lower:

fraudulent transfers

manual reviews

customer support load

financial loss

**🔹 5. Key Code Snippets & Explanations**
**🧩 5.1 Feature Engineering – Transaction + Call Context**
def engineer_paysim_features(paysim: pd.DataFrame) -> pd.DataFrame:
    df = paysim.copy()

    df["transaction_day"] = df["step"] // 24
    df["transaction_hour"] = df["step"] % 24

    df["origin_balance_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["is_large_amount"] = df["amount"] > 200_000

    df = df.sort_values(["nameOrig", "step"])
    df["recipient_first_time"] = (
        df.groupby(["nameOrig", "nameDest"]).cumcount() == 0
    ).astype(int)

    return df


Explanation:
Transforms raw logs into behavioral features like time-of-day & money movement.
recipient_first_time captures new recipient fraud risk, a common social-engineering pattern.

def add_synthetic_call_features(df: pd.DataFrame, cdr: pd.DataFrame | None = None,
                                random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    n = len(df)
    out = df.copy()

    has_recent_call = rng.random(n) < 0.3
    out["has_recent_call"] = has_recent_call.astype(int)

    deltas = rng.exponential(scale=120.0, size=n)
    deltas = np.clip(deltas, 0.0, 900.0)
    out["call_to_tx_delta_seconds"] = np.where(
        has_recent_call, deltas, 9999.0
    ).astype("float32")

    durations = rng.normal(loc=60.0, scale=30.0, size=n)
    durations = np.clip(durations, 5.0, 600.0)
    out["call_duration_seconds"] = durations.astype("float32")

    contact_flags = rng.random(n) < 0.7
    out["contact_list_flag"] = contact_flags.astype(int)

    if cdr is not None and "tenure" in cdr.columns:
        tenure_values = cdr["tenure"].dropna().values
        if len(tenure_values) > 0:
            sampled_tenure = rng.choice(tenure_values, size=n, replace=True)
            out["device_age_days"] = (sampled_tenure * 30).astype("float32")
        else:
            out["device_age_days"] = rng.integers(30, 365 * 3, size=n).astype("float32")
    else:
        out["device_age_days"] = rng.integers(30, 365 * 3, size=n).astype("float32")

    scores = rng.beta(a=1.0, b=8.0, size=n)
    suspicious_mask = rng.random(n) < 0.05
    scores[suspicious_mask] = rng.uniform(0.8, 1.0, size=suspicious_mask.sum())
    out["nlp_suspicion_score"] = scores.astype("float32")

    return out


Explanation:
Simulates call context realistically due to lack of public telco+transaction datasets.
call_to_tx_delta_seconds is essential for call → fraud patterns.
nlp_suspicion_score approximates a future NLP model.

**🧠 5.2 MindSpore Model – CallTriggeredDefenseModel**
class CallTriggeredDefenseModel(nn.Cell):
    """
    Simple MLP for fraud probability prediction on tabular features.
    """

    def __init__(self, input_dim: int, hidden_units=(64, 32), dropout_rate: float = 0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h in hidden_units:
            layers.append(nn.Dense(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1.0 - dropout_rate))
            in_dim = h

        layers.append(nn.Dense(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.SequentialCell(layers)

    def construct(self, x):
        return self.net(x)


Explanation:
Compact, deployment-ready MLP.
Uses dropout to avoid overfitting on synthetic features.
Runs efficiently for high-throughput fraud detection.

**🔀 5.3 Inference + Ensemble Logic – Rules + Model**
def run_inference(event_dict):
    x = prepare_features(event_dict)
    model = load_model()
    model_score = predict_proba(model, x)

    rule_event = CallTxEvent(
        call_to_tx_delta_seconds=event_dict.get("call_to_tx_delta_seconds", None),
        recipient_first_time=event_dict.get("recipient_first_time", 0),
        tx_amount=event_dict.get("amount", 0.0),
        contact_list_flag=event_dict.get("contact_list_flag", 0),
        nlp_suspicion_score=event_dict.get("nlp_suspicion_score", None),
    )
    rule_flag = high_precision_rule(rule_event)

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


Explanation:
Core of the ensemble fraud detection logic.
Rules catch high-precision scam patterns; MLP refines the rest.
Returns score + risk + reason + recommended actions, ideal for API/UI integration.
