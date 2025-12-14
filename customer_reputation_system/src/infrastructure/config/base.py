"""Base configuration settings."""

from typing import Optional


class Settings:
    """Main application settings."""
    
    def __init__(self):
        # Environment
        self.environment = "development"
        self.debug = True
        
        # Application
        self.app_name = "Customer Reputation System"
        self.version = "1.0.0"
        
        # Logging
        self.log_level = "INFO"
        self.log_file: Optional[str] = None
        
        # Security
        self.secret_key = "dev-secret-key-change-in-production"
        
        # API
        self.api_host = "0.0.0.0"
        self.api_port = 8000
        self.api_prefix = "/api/v1"
        
        # Data directories
        self.data_dir = "data"
        self.raw_data_dir = "data/raw"
        self.processed_data_dir = "data/processed"
        self.models_dir = "models"


# Global settings instance
settings = Settings()