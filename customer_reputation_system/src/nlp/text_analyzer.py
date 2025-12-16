"""
Text analyzer for real-time report text analysis
"""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

import mindspore as ms
from mindspore import Tensor, ops

from .preprocessor import TextPreprocessor
from .model import SimpleTextClassifier
from .config import (
    CHECKPOINT_PATH,
    VOCAB_PATH,
    MAX_SEQUENCE_LENGTH,
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


class TextAnalyzer:
    """
    Main text analyzer for report text classification and analysis
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        vocab_path: Optional[Path] = None,
        use_simple_model: bool = True,
    ):
        """
        Initialize text analyzer
        
        Args:
            model_path: Path to model checkpoint
            vocab_path: Path to vocabulary file
            use_simple_model: Use SimpleTextClassifier (faster) vs ReportTextClassifier
        """
        self.model_path = model_path or CHECKPOINT_PATH
        self.vocab_path = vocab_path or VOCAB_PATH
        self.use_simple_model = use_simple_model
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(self.vocab_path)
        
        # Load vocabulary if exists
        if self.vocab_path.exists():
            self.preprocessor.load_vocab()
        else:
            logger.warning(f"Vocabulary not found at {self.vocab_path}. "
                          "Model will need to be trained first.")
        
        # Initialize model
        self.model = None
        self.vocab_size = self.preprocessor.vocab_size or 5000
        
        if self.model_path.exists():
            self._load_model()
        else:
            logger.warning(f"Model checkpoint not found at {self.model_path}. "
                          "Model will need to be trained first.")

    def _load_model(self):
        """Load trained model from checkpoint"""
        try:
            if self.use_simple_model:
                self.model = SimpleTextClassifier(
                    vocab_size=self.vocab_size,
                    embedding_dim=EMBEDDING_DIM,
                    hidden_units=HIDDEN_UNITS,
                    dropout_rate=DROPOUT_RATE,
                )
            else:
                from maifds_governance.nlp.model import ReportTextClassifier
                self.model = ReportTextClassifier(
                    vocab_size=self.vocab_size,
                    embedding_dim=EMBEDDING_DIM,
                    hidden_units=HIDDEN_UNITS,
                    dropout_rate=DROPOUT_RATE,
                )
            
            # Load checkpoint
            param_dict = ms.load_checkpoint(str(self.model_path))
            ms.load_param_into_net(self.model, param_dict)
            self.model.set_train(False)
            
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze report text and return classification results
        
        Args:
            text: Report text (title + description)
            
        Returns:
            Dictionary with analysis results
        """
        if self.model is None:
            logger.warning("Model not loaded. Returning basic features only.")
            return self._analyze_basic_features(text)
        
        try:
            # Preprocess and convert to sequence
            sequence = self.preprocessor.text_to_sequence(
                text, max_length=MAX_SEQUENCE_LENGTH
            )
            
            # Convert to tensor
            x = Tensor([sequence], ms.int32)
            
            # Forward pass
            sentiment_logits, urgency_logits, credibility = self.model(x)
            
            # Process outputs
            # Sentiment
            sentiment_probs = ops.softmax(sentiment_logits, axis=1)
            sentiment_idx = sentiment_probs.argmax(axis=1).asnumpy()[0]
            sentiment = SENTIMENT_CLASSES[sentiment_idx]
            sentiment_confidence = float(sentiment_probs[0, sentiment_idx].asnumpy())
            
            # Urgency
            urgency_probs = ops.softmax(urgency_logits, axis=1)
            urgency_idx = urgency_probs.argmax(axis=1).asnumpy()[0]
            urgency = URGENCY_LEVELS[urgency_idx]
            urgency_confidence = float(urgency_probs[0, urgency_idx].asnumpy())
            
            # Credibility
            credibility_score = float(credibility.asnumpy()[0, 0])
            
            # Combine with basic features
            basic_features = self.preprocessor.extract_text_features(text)
            
            return {
                'sentiment': sentiment,
                'sentiment_confidence': sentiment_confidence,
                'sentiment_distribution': {
                    cls: float(prob.asnumpy()[0])
                    for cls, prob in zip(SENTIMENT_CLASSES, sentiment_probs[0])
                },
                'urgency': urgency,
                'urgency_confidence': urgency_confidence,
                'urgency_distribution': {
                    level: float(prob.asnumpy()[0])
                    for level, prob in zip(URGENCY_LEVELS, urgency_probs[0])
                },
                'credibility_score': credibility_score,
                'text_features': basic_features,
            }
        except Exception as e:
            logger.error(f"Error during text analysis: {e}")
            return self._analyze_basic_features(text)

    def _analyze_basic_features(self, text: str) -> Dict[str, Any]:
        """
        Fallback: analyze text using only basic features (no model)
        
        Args:
            text: Input text
            
        Returns:
            Basic feature analysis
        """
        features = self.preprocessor.extract_text_features(text)
        
        # Simple rule-based sentiment
        if features['negative_word_count'] > 3:
            sentiment = 'negative'
        elif features['negative_word_count'] == 0 and features['word_count'] > 10:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'
        
        # Simple rule-based urgency
        if features['urgency_keyword_count'] >= 3:
            urgency = 'critical'
        elif features['urgency_keyword_count'] >= 2:
            urgency = 'high'
        elif features['urgency_keyword_count'] >= 1:
            urgency = 'medium'
        else:
            urgency = 'low'
        
        # Simple credibility (based on text quality)
        credibility = min(1.0, max(0.0, 
            0.5 + (features['word_count'] / 100) - (features['negative_word_count'] / 10)
        ))
        
        return {
            'sentiment': sentiment,
            'sentiment_confidence': 0.6,  # Lower confidence for rule-based
            'urgency': urgency,
            'urgency_confidence': 0.6,
            'credibility_score': credibility,
            'text_features': features,
            'analysis_method': 'rule_based',
        }

    def analyze_report(self, title: str, description: str) -> Dict[str, Any]:
        """
        Analyze a complete report (title + description)
        
        Args:
            title: Report title
            description: Report description
            
        Returns:
            Combined analysis results
        """
        # Combine title and description
        combined_text = f"{title}. {description}"
        
        # Analyze combined text
        analysis = self.analyze_text(combined_text)
        
        # Add separate analyses
        title_analysis = self.analyze_text(title)
        description_analysis = self.analyze_text(description)
        
        return {
            **analysis,
            'title_analysis': {
                'sentiment': title_analysis.get('sentiment'),
                'urgency': title_analysis.get('urgency'),
            },
            'description_analysis': {
                'sentiment': description_analysis.get('sentiment'),
                'urgency': description_analysis.get('urgency'),
            },
        }

