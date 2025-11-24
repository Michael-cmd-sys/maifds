from pathlib import Path

# Base directory for this feature (call_triggered_defense/)
FEATURE_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = FEATURE_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw dataset paths
PAYSIM_PATH = RAW_DATA_DIR / "paysim.csv"
CDR_PATH = RAW_DATA_DIR / "cdr.csv"
SMS_SPAM_PATH = RAW_DATA_DIR / "sms_spam.csv"

# Processed dataset path (we'll create this later)
TRAINING_TABLE_PATH = PROCESSED_DATA_DIR / "call_tx_training_table.parquet"

# Basic training hyperparameters (can tune later)
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 20
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# To avoid running out of memory in WSL, limit how many PaySim rows we load
MAX_PAYSIM_ROWS = 200_000  # adjust up/down depending on your RAM


# Supervised learning setup
LABEL_COLUMN = "isFraud"

# Input features for the first model version (transaction-only)
FEATURE_COLUMNS = [
    "amount",
    "transaction_day",
    "transaction_hour",
    "origin_balance_delta",
    "dest_balance_delta",
    "is_large_amount",
    "recipient_first_time",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "has_recent_call",
    "call_to_tx_delta_seconds",
    "call_duration_seconds",
    "contact_list_flag",
    "device_age_days",
    "nlp_suspicion_score",
]
