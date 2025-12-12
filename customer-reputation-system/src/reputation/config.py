"""
Configuration for Merchant Reputation Scoring System
"""

# Reputation calculation weights
RATING_WEIGHT = 0.35  # Weight for credibility-weighted ratings
SENTIMENT_WEIGHT = 0.25  # Weight for sentiment analysis
FRAUD_RISK_WEIGHT = 0.20  # Weight for fraud report ratio (inverse)
VOLUME_WEIGHT = 0.10  # Weight for report volume
TIME_DECAY_WEIGHT = 0.10  # Weight for recent activity

# Rating normalization
MIN_RATING = 1.0
MAX_RATING = 5.0
RATING_TO_SCORE_DIVISOR = MAX_RATING  # Normalize rating to 0-1

# Credibility weighting
MIN_CREDIBILITY_FOR_WEIGHT = 0.3  # Minimum credibility to count in weighted average
CREDIBILITY_BOOST_THRESHOLD = 0.7  # High credibility reporters get boost
CREDIBILITY_BOOST_AMOUNT = 0.1  # Boost amount for high credibility

# Sentiment scoring
POSITIVE_SENTIMENT_VALUE = 1.0
NEUTRAL_SENTIMENT_VALUE = 0.5
NEGATIVE_SENTIMENT_VALUE = 0.0

# Fraud risk
FRAUD_REPORT_PENALTY = 0.3  # Penalty for each fraud report (up to max)
MAX_FRAUD_PENALTY = 0.5  # Maximum penalty from fraud reports

# Volume scoring
MIN_REPORTS_FOR_VOLUME = 3  # Minimum reports for volume score
OPTIMAL_REPORT_COUNT = 20  # Optimal number of reports for volume score
MAX_VOLUME_SCORE = 1.0  # Maximum volume score

# Time decay parameters
RECENT_REPORT_DAYS = 30  # Reports within this many days are considered "recent"
TIME_DECAY_HALF_LIFE_DAYS = 90  # Half-life for time decay calculation

# Trend analysis
TREND_WINDOW_DAYS = 90  # Days to look back for trend analysis
TREND_THRESHOLD = 0.1  # Minimum change to be considered a trend

# Initial reputation
INITIAL_REPUTATION = 0.5  # Starting reputation for new merchants

# Reputation update frequency
UPDATE_ON_EVERY_REPORT = True  # Update reputation on every new report

