"""
Training script for report text classifier
"""

import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import nn, Tensor, ops
from pathlib import Path
from typing import Tuple

from src.nlp.preprocessor import TextPreprocessor
from src.nlp.model import SimpleTextClassifier
from src.nlp.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    TRAIN_VAL_SPLIT,
    MAX_SEQUENCE_LENGTH,
    CHECKPOINT_PATH,
    VOCAB_PATH,
    SENTIMENT_CLASSES,
    URGENCY_LEVELS,
    EMBEDDING_DIM,
    HIDDEN_UNITS,
    DROPOUT_RATE,
)
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.logging_config import setup_logger

logger = setup_logger(__name__)


def load_training_data(db_path: Path) -> Tuple[list, list, list]:
    """
    Load reports from database for training
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        Tuple of (texts, sentiment_labels, urgency_labels)
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT title, description, report_type, rating FROM reports")
    rows = cursor.fetchall()
    conn.close()
    
    texts = []
    sentiment_labels = []
    urgency_labels = []
    
    # Map report_type to sentiment (simplified)
    type_to_sentiment = {
        'fraud': 1,  # negative
        'service_issue': 1,  # negative
        'technical': 0,  # neutral
        'other': 0,  # neutral
    }
    
    # Map rating to sentiment (if available)
    rating_to_sentiment = {
        1: 1,  # negative
        2: 1,  # negative
        3: 0,  # neutral
        4: 2,  # positive
        5: 2,  # positive
    }
    
    for row in rows:
        title = row['title'] or ''
        description = row['description'] or ''
        combined = f"{title}. {description}"
        
        if len(combined.strip()) < 10:
            continue
        
        texts.append(combined)
        
        # Determine sentiment
        report_type = row['report_type'] or 'other'
        rating = row['rating']
        
        if rating and rating in rating_to_sentiment:
            sentiment = rating_to_sentiment[rating]
        else:
            sentiment = type_to_sentiment.get(report_type, 0)
        
        sentiment_labels.append(sentiment)
        
        # Determine urgency (simplified: fraud = high, others = medium/low)
        if report_type == 'fraud':
            urgency = 2  # high
        elif report_type == 'service_issue':
            urgency = 1  # medium
        else:
            urgency = 0  # low
        
        urgency_labels.append(urgency)
    
    logger.info(f"Loaded {len(texts)} training samples")
    return texts, sentiment_labels, urgency_labels


def create_dataset(texts: list, sentiment_labels: list, urgency_labels: list,
                   preprocessor: TextPreprocessor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training dataset
    
    Args:
        texts: List of text samples
        sentiment_labels: List of sentiment labels
        urgency_labels: List of urgency labels
        preprocessor: Text preprocessor
        
    Returns:
        Tuple of (sequences, sentiment_labels, urgency_labels) as numpy arrays
    """
    sequences = []
    for text in texts:
        seq = preprocessor.text_to_sequence(text, max_length=MAX_SEQUENCE_LENGTH)
        sequences.append(seq)
    
    return (
        np.array(sequences, dtype=np.int32),
        np.array(sentiment_labels, dtype=np.int32),
        np.array(urgency_labels, dtype=np.int32),
    )


def train():
    """Main training function"""
    ms.set_seed(RANDOM_SEED)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # Load data from database
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from config.settings import DATABASE_PATH
    
    if not DATABASE_PATH.exists():
        logger.error(f"Database not found at {DATABASE_PATH}. "
                     "Please run the report ingestion system first to create sample data.")
        return
    
    logger.info("Loading training data from database...")
    texts, sentiment_labels, urgency_labels = load_training_data(DATABASE_PATH)
    
    if len(texts) < 10:
        logger.warning("Not enough training data. Need at least 10 samples. "
                      "Please submit more reports first.")
        return
    
    # Build vocabulary
    logger.info("Building vocabulary...")
    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(texts, vocab_size=5000)
    preprocessor.save_vocab(VOCAB_PATH)
    
    # Create dataset
    logger.info("Creating dataset...")
    sequences, sentiment_labels, urgency_labels = create_dataset(
        texts, sentiment_labels, urgency_labels, preprocessor
    )
    
    # Train/val split
    n = len(sequences)
    val_size = int(n * TRAIN_VAL_SPLIT)
    train_size = n - val_size
    
    indices = np.random.permutation(n)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train = sequences[train_indices]
    y_sentiment_train = sentiment_labels[train_indices]
    y_urgency_train = urgency_labels[train_indices]
    
    X_val = sequences[val_indices]
    y_sentiment_val = sentiment_labels[val_indices]
    y_urgency_val = urgency_labels[val_indices]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create MindSpore dataset
    train_ds = ms.dataset.NumpySlicesDataset({
        'sequences': X_train,
        'sentiment': y_sentiment_train,
        'urgency': y_urgency_train,
    }, shuffle=True)
    train_ds = train_ds.batch(BATCH_SIZE)
    
    val_ds = ms.dataset.NumpySlicesDataset({
        'sequences': X_val,
        'sentiment': y_sentiment_val,
        'urgency': y_urgency_val,
    }, shuffle=False)
    val_ds = val_ds.batch(BATCH_SIZE)
    
    # Initialize model
    vocab_size = preprocessor.vocab_size
    model = SimpleTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
    )
    
    # Loss functions
    sentiment_loss_fn = nn.CrossEntropyLoss()
    urgency_loss_fn = nn.CrossEntropyLoss()
    credibility_loss_fn = nn.MSELoss()
    
    # Optimizer
    optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.set_train(True)
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_ds.create_dict_iterator():
            sequences = Tensor(batch['sequences'], ms.int32)
            sentiment_targets = Tensor(batch['sentiment'], ms.int32)
            urgency_targets = Tensor(batch['urgency'], ms.int32)
            
            def forward_fn(x, y_sent, y_urg):
                sentiment_logits, urgency_logits, credibility = model(x)
                
                # Losses
                loss_sentiment = sentiment_loss_fn(sentiment_logits, y_sent)
                loss_urgency = urgency_loss_fn(urgency_logits, y_urg)
                
                # Credibility loss (use sentiment as proxy)
                credibility_target = 1.0 - (y_sent.float() / 2.0)  # 0->1.0, 1->0.5, 2->0.0
                loss_credibility = credibility_loss_fn(
                    credibility.squeeze(), credibility_target
                )
                
                total_loss = loss_sentiment + loss_urgency + 0.5 * loss_credibility
                return total_loss, (loss_sentiment, loss_urgency, loss_credibility)
            
            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, aux_losses), grads = grad_fn(sequences, sentiment_targets, urgency_targets)
            optimizer(grads)
            
            total_loss += float(loss.asnumpy())
            num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        # Validation
        model.set_train(False)
        val_loss = 0.0
        val_batches = 0
        correct_sentiment = 0
        correct_urgency = 0
        total_samples = 0
        
        for batch in val_ds.create_dict_iterator():
            sequences = Tensor(batch['sequences'], ms.int32)
            sentiment_targets = Tensor(batch['sentiment'], ms.int32)
            urgency_targets = Tensor(batch['urgency'], ms.int32)
            
            sentiment_logits, urgency_logits, _ = model(sequences)
            
            # Loss
            loss_sentiment = sentiment_loss_fn(sentiment_logits, sentiment_targets)
            loss_urgency = urgency_loss_fn(urgency_logits, urgency_targets)
            val_loss += float((loss_sentiment + loss_urgency).asnumpy())
            val_batches += 1
            
            # Accuracy
            pred_sentiment = sentiment_logits.argmax(axis=1)
            pred_urgency = urgency_logits.argmax(axis=1)
            
            correct_sentiment += (pred_sentiment == sentiment_targets).sum().asnumpy()
            correct_urgency += (pred_urgency == urgency_targets).sum().asnumpy()
            total_samples += len(sentiment_targets)
        
        avg_val_loss = val_loss / max(1, val_batches)
        sentiment_acc = correct_sentiment / max(1, total_samples)
        urgency_acc = correct_urgency / max(1, total_samples)
        
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"train_loss={avg_loss:.4f} - "
            f"val_loss={avg_val_loss:.4f} - "
            f"sentiment_acc={sentiment_acc:.4f} - "
            f"urgency_acc={urgency_acc:.4f}"
        )
    
    # Save model
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ms.save_checkpoint(model, str(CHECKPOINT_PATH))
    logger.info(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()

