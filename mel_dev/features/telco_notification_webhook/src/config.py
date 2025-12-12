from pathlib import Path
import os

# Root of this feature
FEATURE_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = FEATURE_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"

# Where we log incidents + failures
INCIDENT_LOG_PATH = LOG_DIR / "incidents.jsonl"
ERROR_LOG_PATH = LOG_DIR / "webhook_errors.jsonl"

# Webhook configuration (to telco / SIEM / queue gateway)
# In real deployments, these should come from environment variables or secure config.
TELCO_WEBHOOK_URL = os.getenv(
    "TELCO_WEBHOOK_URL",
    "https://webhook.site/092dcb2f-b791-44be-b1f5-291a3bbae1ef"  # placeholder
)

TELCO_WEBHOOK_API_KEY = os.getenv(
    "TELCO_WEBHOOK_API_KEY",
    "CHANGE_ME_IN_PRODUCTION"
)

# Optional: mutual TLS / SSL verification toggle
VERIFY_SSL = os.getenv("TELCO_WEBHOOK_VERIFY_SSL", "true").lower() == "true"

# Timeouts & retry policy
REQUEST_TIMEOUT_SECONDS = float(os.getenv("TELCO_WEBHOOK_TIMEOUT", "5.0"))
MAX_RETRIES = int(os.getenv("TELCO_WEBHOOK_MAX_RETRIES", "3"))

# Metadata about our system, useful for telco
SYSTEM_NAME = os.getenv("FRAUD_PLATFORM_SYSTEM_NAME", "mindspore_fraud_platform")
ENVIRONMENT = os.getenv("FRAUD_PLATFORM_ENV", "dev")
