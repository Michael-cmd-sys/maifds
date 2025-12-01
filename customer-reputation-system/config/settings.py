"""
Configuration settings for the Customer Reputation System
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database settings
DATABASE_PATH = DATABASE_DIR / "reports.db"

# Report validation settings
REPORT_TYPES = ["fraud", "service_issue", "technical", "other"]
PLATFORMS = ["mobile", "web", "api"]

# Text length constraints
MIN_TITLE_LENGTH = 3
MAX_TITLE_LENGTH = 200
MIN_DESCRIPTION_LENGTH = 10
MAX_DESCRIPTION_LENGTH = 5000

# Rating constraints
MIN_RATING = 1
MAX_RATING = 5

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# Create logs directory
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
