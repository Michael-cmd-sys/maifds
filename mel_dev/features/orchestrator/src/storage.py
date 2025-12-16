import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import LOG_DIR, ORCH_LOG_PATH, ORCH_ERROR_LOG_PATH


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    _ensure_log_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_orchestrator_run(request_payload: Dict[str, Any], result: Dict[str, Any]) -> None:
    record = {
        "logged_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "request": request_payload,
        "result": result,
    }
    _append_jsonl(ORCH_LOG_PATH, record)


def log_orchestrator_error(request_payload: Dict[str, Any], error: str) -> None:
    record = {
        "logged_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "request": request_payload,
        "error": error,
    }
    _append_jsonl(ORCH_ERROR_LOG_PATH, record)
