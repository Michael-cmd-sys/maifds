import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from .config import INCIDENT_LOG_PATH, ERROR_LOG_PATH, LOG_DIR


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    _ensure_log_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_incident(payload: Dict[str, Any], response: Dict[str, Any]) -> None:
    record = {
        "logged_at": datetime.utcnow().isoformat() + "Z",
        "payload": payload,
        "response": response,
    }
    append_jsonl(INCIDENT_LOG_PATH, record)


def log_error(error_info: Dict[str, Any]) -> None:
    record = {
        "logged_at": datetime.utcnow().isoformat() + "Z",
        **error_info,
    }
    append_jsonl(ERROR_LOG_PATH, record)
