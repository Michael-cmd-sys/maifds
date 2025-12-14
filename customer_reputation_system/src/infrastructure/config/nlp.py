"""NLP configuration settings."""

class NLPSettings:
    """NLP model and processing configuration."""
    
    def __init__(self):
        # Model settings
        self.model_path = "models/nlp"
        self.batch_size = 32
        self.max_sequence_length = 200
        self.vocab_size = 5000
        self.embedding_dim = 128
        
        # Training settings
        self.epochs = 20
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # Processing settings
        self.enable_fallback = True
        self.confidence_threshold = 0.5


# Global NLP settings instance
nlp_settings = NLPSettings()