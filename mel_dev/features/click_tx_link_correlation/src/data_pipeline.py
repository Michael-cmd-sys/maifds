import hashlib
from typing import Optional, List

import numpy as np
import pandas as pd

from config import (
    TRANSACTIONS_PATH,
    URL_RISK_PATH,
    PHISHING_BLACKLIST_PATH,
    PROCESSED_DATA_DIR,
    TRAINING_TABLE_PATH,
    LABEL_COLUMN,
    MAX_TRANSACTIONS_ROWS,
)


# -------------------------------
# Helper: existence checks
# -------------------------------

def assert_raw_data_exists():
    missing = []
    for path in [TRANSACTIONS_PATH, URL_RISK_PATH]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing required raw datasets:\n"
            + "\n".join(missing)
            + "\n\nExpected files in data/raw/. "
              "See docs/data_download_instructions.md for details."
        )


# -------------------------------
# Helper: column detection
# -------------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_url_column(df: pd.DataFrame) -> str:
    """
    Try to infer which column contains the URL.
    Looks for common names like 'url', 'URL', 'phishing_url', etc.
    """
    candidates = [
        "url", "URL", "Url",
        "phishing_url", "phish_url",
        "full_url", "website", "Website",
        "Domain", "domain", "link", "Link",
    ]
    col = _pick_first_existing(df, candidates)
    if col is not None:
        return col

    # fallback: any column containing 'url'
    for c in df.columns:
        if "url" in c.lower():
            return c

    raise KeyError(
        f"Could not find a URL column in columns: {list(df.columns)}. "
        "Please rename the URL column or adjust find_url_column()."
    )


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a column describing malicious/benign label for URLs.
    """
    candidates = ["label", "Label", "LABEL", "status", "Status", "result", "type", "class", "Category"]
    return _pick_first_existing(df, candidates)


def find_timestamp_column(df: pd.DataFrame) -> str:
    """
    Try to find transaction timestamp column in transactions.csv.
    """
    candidates = [
        "tx_timestamp", "timestamp", "Timestamp", "event_time",
        "time", "Time", "transaction_time", "datetime", "DateTime",
    ]
    col = _pick_first_existing(df, candidates)
    if col is None:
        raise KeyError(
            f"Could not find a timestamp column in transactions.csv. "
            f"Looked for: {candidates}. Actual columns: {list(df.columns)}"
        )
    return col


def find_amount_column(df: pd.DataFrame) -> str:
    """
    Try to find transaction amount/price column.
    """
    candidates = ["amount", "Amount", "price", "Price", "transaction_value", "Payment", "payment_value"]
    col = _pick_first_existing(df, candidates)
    if col is None:
        raise KeyError(
            f"Could not find an amount/price column in transactions.csv. "
            f"Looked for: {candidates}. Actual columns: {list(df.columns)}"
        )
    return col


def find_user_column(df: pd.DataFrame) -> Optional[str]:
    """
    Try to find a user / customer / session id column.
    If none is found, we fall back to a single dummy user.
    """
    candidates = [
        "user_id", "UserID", "userID", "customer_id", "CustomerID",
        "client_id", "session_id", "SessionID", "session", "VisitorID",
    ]
    return _pick_first_existing(df, candidates)


def normalise_url_column(df: pd.DataFrame, col: Optional[str] = None) -> pd.Series:
    """
    Lowercase + strip URLs. If col is None, detect automatically.
    """
    if col is None:
        col = find_url_column(df)
    return df[col].astype(str).str.strip().str.lower()


# -------------------------------
# Loading raw datasets
# -------------------------------

def load_raw_datasets():
    """
    Load transactions, URL risk table and optional phishing blacklist.
    """
    transactions = pd.read_csv(TRANSACTIONS_PATH, nrows=MAX_TRANSACTIONS_ROWS)
    url_risk = pd.read_csv(URL_RISK_PATH)

    if PHISHING_BLACKLIST_PATH.exists():
        phishing_blacklist = pd.read_csv(PHISHING_BLACKLIST_PATH)
    else:
        phishing_blacklist = pd.DataFrame(columns=["url"])

    return {
        "transactions": transactions,
        "url_risk": url_risk,
        "phishing_blacklist": phishing_blacklist,
    }


# -------------------------------
# URL risk table construction
# -------------------------------

def build_url_risk_table(url_risk: pd.DataFrame,
                         phishing_blacklist: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact URL risk table:
      url_clean -> url_hash_numeric, url_risk_score, url_reported_flag
    """
    df = url_risk.copy()

    # 1) URL normalization
    df["url_clean"] = normalise_url_column(df)

    # 2) Risk score from label column (if present)
    label_col = find_label_column(df)
    if label_col is not None:
        df["url_risk_score"] = df[label_col].astype(str).str.lower().isin(
            ["malicious", "phishing", "phish", "bad", "fraud", "1", "true"]
        ).astype("float32")
    else:
        # If no label column, assume all URLs in this table are risky (score 1.0)
        df["url_risk_score"] = 1.0

    # 3) Reported flag from blacklist
    if not phishing_blacklist.empty:
        bl = phishing_blacklist.copy()
        bl["url_clean"] = normalise_url_column(bl)
        bl_set = set(bl["url_clean"].tolist())
        df["url_reported_flag"] = df["url_clean"].isin(bl_set).astype("int32")
    else:
        df["url_reported_flag"] = 0

    # 4) Deterministic hash
    def hash_url(u: str) -> int:
        h = hashlib.sha256(u.encode("utf-8")).hexdigest()
        return int(h[:16], 16)  # first 64 bits as int

    df["url_hash_numeric"] = df["url_clean"].apply(hash_url).astype("int64")

    df_risk = df[[
        "url_clean",
        "url_hash_numeric",
        "url_risk_score",
        "url_reported_flag",
    ]].drop_duplicates()

    return df_risk


# -------------------------------
# Click → transaction features
# -------------------------------

def engineer_click_tx_features(
    transactions: pd.DataFrame,
    url_risk_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create synthetic click→transaction features for each transaction.

    We assume transactions.csv has at least:
      - a timestamp column (auto-detected)
      - an amount/price column (auto-detected)
      - user/session column (optional)
    """

    df = transactions.copy()

    # Detect key columns
    ts_col = find_timestamp_column(df)
    amt_col = find_amount_column(df)
    user_col = find_user_column(df)

    # Rename to standard internal names
    df.rename(columns={ts_col: "tx_timestamp", amt_col: "amount"}, inplace=True)
    if user_col is not None:
        df.rename(columns={user_col: "user_id"}, inplace=True)
    else:
        df["user_id"] = 0  # single-user fallback

    # Parse timestamps
    df["tx_timestamp"] = pd.to_datetime(df["tx_timestamp"])

    # Sample one URL per transaction
    risk_table = url_risk_table.reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(42)

    idx = rng.integers(0, len(risk_table), size=n)
    df["url_clean"] = risk_table["url_clean"].values[idx]
    df["url_hash_numeric"] = risk_table["url_hash_numeric"].values[idx]
    df["url_risk_score"] = risk_table["url_risk_score"].values[idx]
    df["url_reported_flag"] = risk_table["url_reported_flag"].values[idx]

    # Simulate click time before transaction
    deltas = rng.exponential(scale=120.0, size=n)  # mean ≈ 2 minutes
    deltas = np.clip(deltas, 1.0, 3600.0)          # between 1 second and 1 hour
    df["time_between_click_and_tx"] = deltas.astype("float32")
    df["click_timestamp"] = df["tx_timestamp"] - pd.to_timedelta(deltas, unit="s")

    # Clicked recently? (within 5 minutes)
    df["clicked_recently"] = (df["time_between_click_and_tx"] <= 300.0).astype("int32")

    # Temporal covariates
    df["tx_hour"] = df["tx_timestamp"].dt.hour.astype("int32")
    df["tx_dayofweek"] = df["tx_timestamp"].dt.dayofweek.astype("int32")

    # Behaviour: simple device/user counts and time since last tx
    df = df.sort_values(["user_id", "tx_timestamp"])
    df["device_click_count_1d"] = (
        df.groupby("user_id").cumcount() + 1
    ).astype("int32")
    df["user_click_count_1d"] = df["device_click_count_1d"]

    df["time_since_last_tx_seconds"] = (
        df.groupby("user_id")["tx_timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(99999.0)
        .astype("float32")
    )

    # Label: true fraud if column exists, else synthetic label
    if "is_fraud" in df.columns:
        df[LABEL_COLUMN] = df["is_fraud"].astype("int32")
    else:
        risky_url = df["url_risk_score"] > 0.5
        recent = df["clicked_recently"] == 1
        large_amount = df["amount"] > df["amount"].median()
        df[LABEL_COLUMN] = (risky_url & recent & large_amount).astype("int32")

    return df


# -------------------------------
# Main builder
# -------------------------------

def build_training_table():
    """
    Build and save the processed training table for
    Click → Transaction Link Correlation feature.
    """
    assert_raw_data_exists()
    datasets = load_raw_datasets()

    tx = datasets["transactions"]
    url_risk = datasets["url_risk"]
    phishing_blacklist = datasets["phishing_blacklist"]

    url_risk_table = build_url_risk_table(url_risk, phishing_blacklist)
    training_df = engineer_click_tx_features(tx, url_risk_table)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    training_df.to_parquet(TRAINING_TABLE_PATH, index=False)

    print("Training table shape:", training_df.shape)
    print("Saved to:", TRAINING_TABLE_PATH)

    return training_df


if __name__ == "__main__":
    build_training_table()
