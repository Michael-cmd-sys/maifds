import pandas as pd
import numpy as np
from pathlib import Path

from .config import (
    PAYSIM_PATH,
    CDR_PATH,
    SMS_SPAM_PATH,
    PROCESSED_DATA_DIR,
    TRAINING_TABLE_PATH,
    MAX_PAYSIM_ROWS,
)


def load_raw_datasets():
    """
    Load the three raw datasets:
    - PaySim transactions
    - Telco churn/call-behaviour dataset (used as proxy for call behaviour)
    - SMS spam dataset (for optional NLP suspicion score later)
    """
    paysim = pd.read_csv(PAYSIM_PATH, nrows=MAX_PAYSIM_ROWS)
    print(f"Loaded PaySim with {len(paysim):,} rows (limited by MAX_PAYSIM_ROWS).")
    cdr = pd.read_csv(CDR_PATH)
    sms_spam = pd.read_csv(SMS_SPAM_PATH, encoding="latin-1")  # common for this dataset

    return {
        "paysim": paysim,
        "cdr": cdr,
        "sms_spam": sms_spam,
    }


def engineer_paysim_features(paysim: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic fraud-related features on top of the raw PaySim dataset.

    We do:
    - transaction_day, transaction_hour (derived from `step`)
    - balance deltas for origin and destination
    - is_large_amount flag
    - recipient_first_time flag (is this the first time this user sends to this recipient?)
    """
    df = paysim.copy()

    # Time features from `step` (each step is 1 hour in the simulation)
    df["transaction_day"] = df["step"] // 24
    df["transaction_hour"] = df["step"] % 24

    # Balance change features
    df["origin_balance_delta"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # Simple large-amount flag (you can tune this threshold later)
    df["is_large_amount"] = df["amount"] > 200000  # example threshold

    # Recipient-first-time: for each (sender, recipient) pair, mark if this is the first tx
    df = df.sort_values(["nameOrig", "step"])
    df["recipient_first_time"] = (
        df.groupby(["nameOrig", "nameDest"]).cumcount() == 0
    ).astype(int)

    return df


def add_synthetic_call_features(
    df: pd.DataFrame,
    cdr: pd.DataFrame | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Add synthetic call/context features for Call → Tx defense.

    We simulate:
    - has_recent_call (0/1)
    - call_to_tx_delta_seconds (seconds since last call; large value if no call)
    - call_duration_seconds
    - contact_list_flag (0/1)
    - device_age_days (from churn 'tenure' if available, else random)
    - nlp_suspicion_score (0–1, with a few high-risk cases)
    """
    rng = np.random.default_rng(random_seed)
    n = len(df)
    out = df.copy()

    # 1) Whether there was a recent call before the transaction
    has_recent_call = rng.random(n) < 0.3  # 30% of tx have a recent call
    out["has_recent_call"] = has_recent_call.astype(int)

    # 2) Time from call to transaction (only where has_recent_call = 1)
    deltas = rng.exponential(scale=120.0, size=n)  # mean ~120s
    deltas = np.clip(deltas, 0.0, 900.0)           # cap at 15 minutes
    out["call_to_tx_delta_seconds"] = np.where(
        has_recent_call,
        deltas,
        9999.0,  # large value when no recent call
    ).astype("float32")

    # 3) Call duration (in seconds)
    durations = rng.normal(loc=60.0, scale=30.0, size=n)  # mean 1 min
    durations = np.clip(durations, 5.0, 600.0)
    out["call_duration_seconds"] = durations.astype("float32")

    # 4) Contact list flag: most callers are in contacts
    contact_flags = rng.random(n) < 0.7
    out["contact_list_flag"] = contact_flags.astype(int)

    # 5) Device age in days – use churn 'tenure' if available, else random
    if cdr is not None and "tenure" in cdr.columns:
        tenure_values = cdr["tenure"].dropna().values
        if len(tenure_values) > 0:
            sampled_tenure = rng.choice(tenure_values, size=n, replace=True)
            out["device_age_days"] = (sampled_tenure * 30).astype("float32")
        else:
            out["device_age_days"] = rng.integers(30, 365 * 3, size=n).astype("float32")
    else:
        out["device_age_days"] = rng.integers(30, 365 * 3, size=n).astype("float32")

    # 6) NLP suspicion score – mostly low, some very high
    scores = rng.beta(a=1.0, b=8.0, size=n)  # mostly near 0
    suspicious_mask = rng.random(n) < 0.05   # ~5% high-risk
    scores[suspicious_mask] = rng.uniform(0.8, 1.0, size=suspicious_mask.sum())
    out["nlp_suspicion_score"] = scores.astype("float32")

    return out



def build_initial_training_table():
    """
    Build the initial training table for the Call Triggered Defense model.

    Steps:
      - Load raw datasets
      - Engineer core PaySim features
      - Add synthetic call/context features
      - Save a sampled training table to data/processed/
    """
    datasets = load_raw_datasets()
    paysim = datasets["paysim"]
    cdr = datasets["cdr"]

    # 1) Transaction-based features
    paysim_fe = engineer_paysim_features(paysim)

    # 2) Add synthetic call/context features
    paysim_with_calls = add_synthetic_call_features(paysim_fe, cdr=cdr, random_seed=42)

    # 3) Sample for faster experimentation
    if len(paysim_with_calls) > 50000:
        sample = paysim_with_calls.sample(n=50000, random_state=42)
    else:
        sample = paysim_with_calls

    # 4) Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(TRAINING_TABLE_PATH, index=False)

    return sample




if __name__ == "__main__":
    # Quick manual test: run this file to check that paths + loading works.
    df = build_initial_training_table()
    print("Training table shape:", df.shape)
    print("Saved to:", TRAINING_TABLE_PATH)
