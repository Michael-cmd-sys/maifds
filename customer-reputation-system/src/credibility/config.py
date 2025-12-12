"""
Configuration for Reporter Credibility System
"""

# Credibility calculation weights
TEXT_CREDIBILITY_WEIGHT = 0.4  # Weight for NLP text credibility scores
CONSISTENCY_WEIGHT = 0.25  # Weight for report consistency
VERIFICATION_WEIGHT = 0.25  # Weight for verified reports
TIME_DECAY_WEIGHT = 0.1  # Weight for recent activity

# Time decay parameters
RECENT_REPORT_DAYS = 30  # Reports within this many days are considered "recent"
TIME_DECAY_HALF_LIFE_DAYS = 90  # Half-life for time decay calculation

# Consistency parameters
MIN_REPORTS_FOR_CONSISTENCY = 3  # Minimum reports needed to calculate consistency
CONSISTENCY_SIMILARITY_THRESHOLD = 0.7  # Similarity threshold for consistent reports

# Verification parameters
VERIFICATION_BOOST = 0.2  # Boost to credibility for verified reports
MIN_VERIFIED_FOR_BOOST = 1  # Minimum verified reports to get boost

# Initial credibility
INITIAL_CREDIBILITY = 0.5  # Starting credibility for new reporters

# Credibility update frequency
UPDATE_ON_EVERY_REPORT = True  # Update credibility on every new report
BATCH_UPDATE_INTERVAL_HOURS = 24  # If not updating every report, batch update interval

