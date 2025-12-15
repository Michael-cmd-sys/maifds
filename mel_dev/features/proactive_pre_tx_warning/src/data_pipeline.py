import numpy as np
import pandas as pd

from .config import (
    PAYSIM_PATH,
    CDR_PATH,
    URL_RISK_PATH,
    PHISHING_BLACKLIST_PATH,
    PROCESSED_DATA_DIR,
    TRAINING_TABLE_PATH,
    LABEL_COLUMN,
    FEATURE_COLUMNS,
    MAX_PAYSIM_ROWS,
    MAX_USERS,
    RANDOM_SEED,
)


# -------------------------
# Basic checks & loading
# -------------------------

def assert_raw_data_exists():
    missing = []
    for path in [PAYSIM_PATH, CDR_PATH, URL_RISK_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing raw datasets for Proactive Pre-Tx Warning:\n"
            + "\n".join(missing)
            + "\n\nExpected files in data/raw/:\n"
              "- paysim.csv\n- cdr.csv\n- url_risk.csv\n"
              "Optional: phishing_blacklist.csv\n"
        )


def load_raw_datasets():
    """
    Load the raw datasets needed for user-level risk cohort generation.
    """
    paysim = pd.read_csv(PAYSIM_PATH, nrows=MAX_PAYSIM_ROWS)
    cdr = pd.read_csv(CDR_PATH)
    url_risk = pd.read_csv(URL_RISK_PATH)

    if PHISHING_BLACKLIST_PATH.exists():
        phishing_blacklist = pd.read_csv(PHISHING_BLACKLIST_PATH)
    else:
        phishing_blacklist = pd.DataFrame(columns=["url"])

    return {
        "paysim": paysim,
        "cdr": cdr,
        "url_risk": url_risk,
        "phishing_blacklist": phishing_blacklist,
    }


# -------------------------
# User aggregates (PaySim)
# -------------------------

def build_user_behavior_table(paysim: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PaySim transactions to user-level behaviour.

    We use:
      - tx_count_total
      - avg_tx_amount
      - max_tx_amount
      - historical_fraud_flag (ever had fraud)
    """

    # ensure expected columns
    required_cols = ["nameOrig", "amount"]
    for col in required_cols:
        if col not in paysim.columns:
            raise KeyError(
                f"Expected column '{col}' in paysim.csv. "
                f"Actual columns: {list(paysim.columns)}"
            )

    if "isFraud" not in paysim.columns:
        paysim["isFraud"] = 0

    user_grp = (
        paysim.groupby("nameOrig")
        .agg(
            tx_count_total=("amount", "size"),
            avg_tx_amount=("amount", "mean"),
            max_tx_amount=("amount", "max"),
            historical_fraud_flag=("isFraud", "max"),
        )
        .reset_index()
        .rename(columns={"nameOrig": "user_id"})
    )

    # limit number of users for development
    if len(user_grp) > MAX_USERS:
        user_grp = user_grp.sample(n=MAX_USERS, random_state=RANDOM_SEED).reset_index(drop=True)

    return user_grp


# -------------------------
# Device age (Telco proxy)
# -------------------------

def attach_device_age(user_df: pd.DataFrame, cdr: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Use Telco churn 'tenure' as a proxy for device age / account age.
    We randomly assign device_age_days to each user from the churn dataset.
    """
    if "tenure" not in cdr.columns:
        # fallback: create synthetic tenure
        cdr["tenure"] = rng.integers(1, 36, size=len(cdr))

    # convert tenure months -> days (approx)
    tenure_days = (cdr["tenure"].astype(float) * 30.0).values
    if len(tenure_days) == 0:
        tenure_days = np.full(shape=len(user_df), fill_value=180.0)  # 6 months default

    sampled_ages = rng.choice(tenure_days, size=len(user_df), replace=True)
    user_df["device_age_days"] = sampled_ages.astype("float32")
    user_df["is_new_device"] = (user_df["device_age_days"] <= 30.0).astype("int32")

    return user_df


# -------------------------
# URL risk & campaign
# -------------------------

def normalise_url_column(df: pd.DataFrame) -> pd.Series:
    """
    Try to find a URL-like column and normalize it (lowercase + strip).
    """
    # heuristics
    for col in df.columns:
        if "url" in col.lower() or col.lower() in ["website", "domain"]:
            return df[col].astype(str).str.strip().str.lower()
    # fallback: first column
    return df.iloc[:, 0].astype(str).str.strip().str.lower()


def compute_global_campaign_intensity(url_risk: pd.DataFrame, phishing_blacklist: pd.DataFrame) -> float:
    """
    Compute a simple global scam campaign intensity score based on
    how many URLs are malicious or in blacklist.
    """
    df = url_risk.copy()
    df["url_clean"] = normalise_url_column(df)

    # detect a label-like column
    label_col = None
    for candidate in ["label", "Label", "status", "type", "result", "class"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        is_risky = df[label_col].astype(str).str.lower().isin(
            ["malicious", "phishing", "phish", "bad", "fraud", "1", "true"]
        )
        risky_frac = is_risky.mean()
    else:
        risky_frac = 0.5  # assume half for synthetic fallback

    # blacklist contribution
    if not phishing_blacklist.empty:
        bl = phishing_blacklist.copy()
        bl["url_clean"] = normalise_url_column(bl)
        bl_count = bl["url_clean"].nunique()
    else:
        bl_count = 0

    # normalize blacklist count
    bl_score = min(bl_count / 1000.0, 1.0)

    # simple combined intensity 0–1
    intensity = float(max(0.0, min(risky_frac * 0.7 + bl_score * 0.3, 1.0)))
    return intensity


def attach_campaign_features(
    user_df: pd.DataFrame,
    url_risk: pd.DataFrame,
    phishing_blacklist: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Simulate per-user recent risky clicks, scam calls and campaign exposure.

    Outputs:
      - recent_risky_clicks_7d
      - recent_scam_calls_7d
      - scam_campaign_intensity (per user, centered around global level)
      - is_in_campaign_cohort
      - user_risk_score (composite)
    """
    global_intensity = compute_global_campaign_intensity(url_risk, phishing_blacklist)

    n_users = len(user_df)

    # Simulate clicks and scam calls (Poisson)
    # scale lambda by global campaign intensity
    lam_clicks = 1.0 + 4.0 * global_intensity   # between ~1 and 5
    lam_calls = 0.5 + 2.0 * global_intensity    # between ~0.5 and 2.5

    user_df["recent_risky_clicks_7d"] = rng.poisson(lam=lam_clicks, size=n_users).astype("int32")
    user_df["recent_scam_calls_7d"] = rng.poisson(lam=lam_calls, size=n_users).astype("int32")

    # Local campaign intensity per user: global + some noise
    noise = rng.normal(loc=0.0, scale=0.1, size=n_users)
    user_df["scam_campaign_intensity"] = np.clip(global_intensity + noise, 0.0, 1.0).astype("float32")

    # In-cohort if they had any risky click or scam call recently
    user_df["is_in_campaign_cohort"] = (
        (user_df["recent_risky_clicks_7d"] > 0) | (user_df["recent_scam_calls_7d"] > 0)
    ).astype("int32")

    # Composite user risk score (0–1) from:
    # - campaign cohort
    # - historical fraud
    # - new device
    # - recent risky activity
    # - campaign intensity
    user_df["user_risk_score"] = (
        0.3 * user_df["is_in_campaign_cohort"].astype("float32")
        + 0.2 * user_df["historical_fraud_flag"].astype("float32")
        + 0.2 * user_df["is_new_device"].astype("float32")
        + 0.15 * (user_df["recent_risky_clicks_7d"] > 0).astype("float32")
        + 0.15 * (user_df["recent_scam_calls_7d"] > 0).astype("float32")
    )

    # clip to [0, 1]
    user_df["user_risk_score"] = np.clip(user_df["user_risk_score"], 0.0, 1.0).astype("float32")

    return user_df


# -------------------------
# Label: who should be warned?
# -------------------------

def attach_warning_label(user_df: pd.DataFrame) -> pd.DataFrame:
    """
    Define a synthetic label "should_warn" based on user_risk_score
    and campaign membership.

    In a real system this would be derived from uplift modeling
    based on actual fraud outcomes after warnings vs no warnings.
    """
    # threshold can be tuned
    warn_threshold = 0.5

    user_df[LABEL_COLUMN] = (
        (user_df["user_risk_score"] >= warn_threshold)
        & (user_df["is_in_campaign_cohort"] == 1)
    ).astype("int32")

    return user_df


# -------------------------
# Main builder
# -------------------------

def build_training_table():
    """
    Build and save the user-level training table for the
    Proactive Pre-Transaction Warning model.
    """
    assert_raw_data_exists()
    datasets = load_raw_datasets()

    paysim = datasets["paysim"]
    cdr = datasets["cdr"]
    url_risk = datasets["url_risk"]
    phishing_blacklist = datasets["phishing_blacklist"]

    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Aggregate transactions -> user behaviour
    user_df = build_user_behavior_table(paysim)

    # 2) Attach device age / new device flag
    user_df = attach_device_age(user_df, cdr, rng)

    # 3) Attach campaign exposure + clicks/calls
    user_df = attach_campaign_features(user_df, url_risk, phishing_blacklist, rng)

    # 4) Attach label "should_warn"
    user_df = attach_warning_label(user_df)

    # 5) Ensure all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in user_df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    user_df.to_parquet(TRAINING_TABLE_PATH, index=False)

    print("Proactive warning training table shape:", user_df.shape)
    print("Saved to:", TRAINING_TABLE_PATH)

    return user_df


if __name__ == "__main__":
    build_training_table()
