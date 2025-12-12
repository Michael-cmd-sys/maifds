"""
Configuration for NLP text analysis module
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model paths
MODEL_DIR = PROJECT_ROOT / "src" / "nlp" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "report_text_classifier.ckpt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
FEATURE_STATS_PATH = MODEL_DIR / "feature_stats.npy"

# Training configuration
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.15

# Text processing
MAX_SEQUENCE_LENGTH = 200  # Max tokens per text
MIN_WORD_LENGTH = 2
VOCAB_SIZE = 5000  # Most frequent words

# Feature extraction
SENTIMENT_CLASSES = ["positive", "negative", "neutral"]
URGENCY_LEVELS = ["low", "medium", "high", "critical"]

# Model architecture
EMBEDDING_DIM = 128
HIDDEN_UNITS = (128, 64)
DROPOUT_RATE = 0.3

