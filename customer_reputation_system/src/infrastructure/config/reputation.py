"""Reputation system configuration settings."""

class ReputationSettings:
    """Merchant reputation calculation configuration."""
    
    def __init__(self):
        # Scoring weights (must sum to 1.0)
        self.weighted_rating_weight = 0.35
        self.sentiment_score_weight = 0.25
        self.fraud_risk_weight = 0.20
        self.volume_score_weight = 0.10
        self.time_decay_weight = 0.10
        
        # Credibility weighting
        self.min_credibility_threshold = 0.3
        self.high_credibility_threshold = 0.7
        self.high_credibility_boost = 0.1
        
        # Volume scoring
        self.optimal_report_count = 20
        self.min_reports_for_scoring = 3
        
        # Fraud penalty
        self.max_fraud_penalty = 0.5
        self.fraud_report_threshold = 0.1
        
        # Time decay settings
        self.recent_report_days = 30
        self.half_life_days = 90
        
        # Trend analysis
        self.trend_window_days = 30
        self.trend_threshold = 0.05


# Global reputation settings instance
reputation_settings = ReputationSettings()