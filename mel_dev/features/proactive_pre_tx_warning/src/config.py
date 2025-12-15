from pathlib import Path
import os

# Root folder for THIS feature
FEATURE_ROOT = Path(__file__).resolve().parents[1]

# Data dirs
DATA_DIR = FEATURE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CKPT_PATH = os.path.join(BASE_DIR, "proactive_warning_mlp.ckpt")
MEAN_PATH = os.path.join(BASE_DIR, "feature_mean.npy")
STD_PATH  = os.path.join(BASE_DIR, "feature_std.npy")

# Raw data paths (reusing files from previous features)
PAYSIM_PATH = RAW_DATA_DIR / "paysim.csv"                 # PaySim transactions
CDR_PATH = RAW_DATA_DIR / "cdr.csv"                       # Telco churn/call proxy
URL_RISK_PATH = RAW_DATA_DIR / "url_risk.csv"             # benign/malicious URLs
PHISHING_BLACKLIST_PATH = RAW_DATA_DIR / "phishing_blacklist.csv"  # optional

# Processed training table
TRAINING_TABLE_PATH = PROCESSED_DATA_DIR / "proactive_warning_training_table.parquet"

# Label
LABEL_COLUMN = "should_warn"

# Feature columns for the uplift/risk model
FEATURE_COLUMNS = [
    "recent_risky_clicks_7d",
    "recent_scam_calls_7d",
    "scam_campaign_intensity",
    "device_age_days",
    "is_new_device",
    "tx_count_total",
    "avg_tx_amount",
    "max_tx_amount",
    "historical_fraud_flag",
    "is_in_campaign_cohort",
    "user_risk_score",
]

# Training hyperparameters (for later train.py)
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# Limits to keep memory reasonable
MAX_PAYSIM_ROWS = 500_000
MAX_USERS = 50_000
