"""
Text preprocessing utilities for NLP analysis
"""

import re
from typing import List, Dict, Optional
from collections import Counter
import json
from pathlib import Path

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from customer_reputation_system.config.logging_config import setup_logger

logger = setup_logger(__name__)


class TextPreprocessor:
    """Handles text preprocessing and tokenization"""

    def __init__(self, vocab_path: Optional[Path] = None):
        """
        Initialize preprocessor
        
        Args:
            vocab_path: Path to vocabulary file (optional)
        """
        self.vocab_path = vocab_path
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        if vocab_path and vocab_path.exists():
            self.load_vocab()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = self.clean_text(text)
        # Simple word tokenization
        tokens = text.split()
        # Filter short tokens
        tokens = [t for t in tokens if len(t) >= 2]
        return tokens

    def build_vocab(self, texts: List[str], vocab_size: int = 5000) -> Dict[str, int]:
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text samples
            vocab_size: Maximum vocabulary size
            
        Returns:
            Word to index mapping
        """
        logger.info(f"Building vocabulary from {len(texts)} texts")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Get most frequent words
        most_common = word_counts.most_common(vocab_size - 2)  # Reserve for PAD and UNK
        
        # Build vocabulary
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        
        for word, _ in most_common:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        # Build reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        logger.info(f"Vocabulary built with {self.vocab_size} words")
        return self.word_to_idx

    def text_to_sequence(self, text: str, max_length: int = 200) -> List[int]:
        """
        Convert text to sequence of word indices
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            List of word indices
        """
        tokens = self.tokenize(text)
        sequence = []
        
        for token in tokens[:max_length]:
            idx = self.word_to_idx.get(token, self.word_to_idx.get('<UNK>', 1))
            sequence.append(idx)
        
        # Pad or truncate to max_length
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))
        else:
            sequence = sequence[:max_length]
        
        return sequence

    def save_vocab(self, vocab_path: Path):
        """Save vocabulary to file"""
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, 'w') as f:
            json.dump(self.word_to_idx, f, indent=2)
        logger.info(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self):
        """Load vocabulary from file"""
        if not self.vocab_path or not self.vocab_path.exists():
            logger.warning("Vocabulary file not found")
            return
        
        with open(self.vocab_path, 'r') as f:
            self.word_to_idx = json.load(f)
        
        # Convert string keys to int for indices
        self.word_to_idx = {k: int(v) if isinstance(v, str) and v.isdigit() else v 
                           for k, v in self.word_to_idx.items()}
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        logger.info(f"Vocabulary loaded with {self.vocab_size} words")

    def extract_text_features(self, text: str) -> Dict[str, float]:
        """
        Extract basic text features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of features
        """
        tokens = self.tokenize(text)
        
        features = {
            'word_count': len(tokens),
            'char_count': len(text),
            'avg_word_length': sum(len(t) for t in tokens) / max(len(tokens), 1),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        }
        
        # Urgency indicators
        urgency_keywords = ['urgent', 'immediately', 'asap', 'critical', 'emergency', 
                          'fraud', 'stolen', 'hacked', 'unauthorized']
        features['urgency_keyword_count'] = sum(1 for word in tokens 
                                                if word in urgency_keywords)
        
        # Negative sentiment indicators
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'disappointed',
                         'angry', 'frustrated', 'scam', 'fraud', 'stolen']
        features['negative_word_count'] = sum(1 for word in tokens 
                                             if word in negative_words)
        
        return features

