"""Credibility system configuration settings."""

class CredibilitySettings:
    """Reporter credibility calculation configuration."""
    
    def __init__(self):
        # Scoring weights (must sum to 1.0)
        self.text_credibility_weight = 0.4
        self.consistency_weight = 0.25
        self.verification_rate_weight = 0.25
        self.time_decay_weight = 0.1
        
        # Thresholds
        self.min_reports_for_consistency = 3
        self.credibility_threshold = 0.3
        self.high_credibility_threshold = 0.7
        
        # Boosts
        self.verification_boost = 0.2
        self.high_credibility_boost = 0.1
        
        # Time decay settings
        self.recent_report_days = 30
        self.half_life_days = 90


# Global credibility settings instance
credibility_settings = CredibilitySettings()