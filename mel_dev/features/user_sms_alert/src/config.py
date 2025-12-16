from pathlib import Path
import os

FEATURE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = FEATURE_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"

SMS_SENT_LOG_PATH = LOG_DIR / "sms_sent.jsonl"
SMS_ERROR_LOG_PATH = LOG_DIR / "sms_errors.jsonl"

MOOLRE_SMS_BASE_URL = os.getenv("MOOLRE_SMS_BASE_URL", "https://api.moolre.com")
MOOLRE_SMS_SEND_PATH = os.getenv("MOOLRE_SMS_SEND_PATH", "/open/sms/send")
MOOLRE_SMS_SEND_URL = MOOLRE_SMS_BASE_URL.rstrip("/") + MOOLRE_SMS_SEND_PATH

MOOLRE_X_API_VASKEY = os.getenv("MOOLRE_X_API_VASKEY", "")
MOOLRE_SENDER_ID = os.getenv("MOOLRE_SENDER_ID", "")

VERIFY_SSL = os.getenv("MOOLRE_VERIFY_SSL", "true").lower() == "true"
REQUEST_TIMEOUT_SECONDS = float(os.getenv("MOOLRE_TIMEOUT_SECONDS", "8"))
MAX_RETRIES = int(os.getenv("MOOLRE_MAX_RETRIES", "3"))
