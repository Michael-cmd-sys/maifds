import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .config import LOG_DIR, SMS_SENT_LOG_PATH, SMS_ERROR_LOG_PATH

def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    _ensure_log_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_sms_sent(payload: Dict[str, Any], response: Dict[str, Any]) -> None:
    record = {
        "logged_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "type": "sms_sent",
        "payload": payload,
        "response": response,
    }
    append_jsonl(SMS_SENT_LOG_PATH, record)

def log_sms_error(error_info: Dict[str, Any]) -> None:
    record = {
        "logged_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "type": "sms_error",
        **error_info,
    }
    append_jsonl(SMS_ERROR_LOG_PATH, record)
