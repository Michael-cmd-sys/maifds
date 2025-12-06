from pathlib import Path

# Root folder for THIS feature (click_tx_link_correlation)
FEATURE_ROOT = Path(__file__).resolve().parents[1]

# Data dirs
DATA_DIR = FEATURE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data paths (you already created these files)
TRANSACTIONS_PATH = RAW_DATA_DIR / "transactions.csv"
URL_RISK_PATH = RAW_DATA_DIR / "url_risk.csv"
PHISHING_BLACKLIST_PATH = RAW_DATA_DIR / "phishing_blacklist.csv"  # optional

# Processed training table
TRAINING_TABLE_PATH = PROCESSED_DATA_DIR / "click_tx_training_table.parquet"

# Supervised label
LABEL_COLUMN = "is_fraud_tx"

# Features for the model (we'll build these in the pipeline)
FEATURE_COLUMNS = [
    # Transaction behaviour
    "amount",
    "time_since_last_tx_seconds",
    "tx_hour",
    "tx_dayofweek",

    # Click / URL risk
    "url_hash_numeric",
    "url_risk_score",
    "url_reported_flag",
    "clicked_recently",
    "time_between_click_and_tx",

    # Device / user activity
    "device_click_count_1d",
    "user_click_count_1d",
]

# Training hyperparameters (used later in train.py)
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# To avoid OOM while experimenting
MAX_TRANSACTIONS_ROWS = 200_000
