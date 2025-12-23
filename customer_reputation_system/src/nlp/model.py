"""
MindSpore model for report text classification
"""

import mindspore as ms
from mindspore import nn, Tensor
import numpy as np

from customer_reputation_system.src.nlp.config import (
    EMBEDDING_DIM,
    HIDDEN_UNITS,
    DROPOUT_RATE,
    MAX_SEQUENCE_LENGTH,
)


class ReportTextClassifier(nn.Cell):
    """
    Text classification model using MindSpore for report analysis.
    Combines embedding + LSTM/GRU + MLP for multi-task learning:
    - Sentiment classification (positive/negative/neutral)
    - Urgency detection (low/medium/high/critical)
    - Credibility score (0-1)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_units: tuple = HIDDEN_UNITS,
        dropout_rate: float = DROPOUT_RATE,
        num_sentiment_classes: int = 3,
        num_urgency_levels: int = 4,
    ):
        """
        Initialize model
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_units: Hidden layer sizes for MLP
            dropout_rate: Dropout rate
            num_sentiment_classes: Number of sentiment classes
            num_urgency_levels: Number of urgency levels
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_sentiment_classes = num_sentiment_classes
        self.num_urgency_levels = num_urgency_levels
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_units[0] // 2,  # Use half for bidirectional
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        lstm_output_dim = hidden_units[0]  # Bidirectional doubles the output
        
        # MLP layers for feature extraction
        mlp_layers = []
        in_dim = lstm_output_dim
        
        for h in hidden_units[1:]:
            mlp_layers.append(nn.Dense(in_dim, h))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(keep_prob=1.0 - dropout_rate))
            in_dim = h
        
        self.mlp = nn.SequentialCell(mlp_layers)
        
        # Task-specific heads
        # Sentiment classification
        self.sentiment_head = nn.Dense(in_dim, num_sentiment_classes)
        
        # Urgency classification
        self.urgency_head = nn.Dense(in_dim, num_urgency_levels)
        
        # Credibility score (regression)
        self.credibility_head = nn.SequentialCell([
            nn.Dense(in_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        ])

    def construct(self, x: Tensor) -> tuple:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Tuple of (sentiment_logits, urgency_logits, credibility_score)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        # Take the last output (or use pooling)
        # Using mean pooling over sequence
        pooled = lstm_out.mean(axis=1)  # (batch, lstm_hidden)
        
        # MLP
        features = self.mlp(pooled)
        
        # Task heads
        sentiment_logits = self.sentiment_head(features)
        urgency_logits = self.urgency_head(features)
        credibility = self.credibility_head(features)
        
        return sentiment_logits, urgency_logits, credibility


class SimpleTextClassifier(nn.Cell):
    """
    Simplified version using only MLP for faster training/testing.
    Uses bag-of-words approach with embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_units: tuple = HIDDEN_UNITS,
        dropout_rate: float = DROPOUT_RATE,
        num_sentiment_classes: int = 3,
        num_urgency_levels: int = 4,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MLP layers
        layers = []
        in_dim = embedding_dim
        
        for h in hidden_units:
            layers.append(nn.Dense(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1.0 - dropout_rate))
            in_dim = h
        
        self.mlp = nn.SequentialCell(layers)
        
        # Task heads
        self.sentiment_head = nn.Dense(in_dim, num_sentiment_classes)
        self.urgency_head = nn.Dense(in_dim, num_urgency_levels)
        self.credibility_head = nn.SequentialCell([
            nn.Dense(in_dim, 1),
            nn.Sigmoid()
        ])

    def construct(self, x: Tensor) -> tuple:
        """
        Forward pass with bag-of-words approach
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Tuple of (sentiment_logits, urgency_logits, credibility_score)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # Mean pooling (bag-of-words style)
        pooled = embedded.mean(axis=1)  # (batch, embedding_dim)
        
        # MLP
        features = self.mlp(pooled)
        
        # Task heads
        sentiment_logits = self.sentiment_head(features)
        urgency_logits = self.urgency_head(features)
        credibility = self.credibility_head(features)
        
        return sentiment_logits, urgency_logits, credibility

