from pathlib import Path
import os

FEATURE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = FEATURE_ROOT / "data"
LOG_DIR = DATA_DIR / "logs"

ORCH_LOG_PATH = LOG_DIR / "orchestrator_runs.jsonl"
ORCH_ERROR_LOG_PATH = LOG_DIR / "orchestrator_errors.jsonl"
