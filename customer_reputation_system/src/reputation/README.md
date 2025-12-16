# Feature 4: Merchant Reputation Scoring

A comprehensive system for calculating merchant reputation scores using credibility-weighted ratings and multi-factor analysis.

## Overview

The Merchant Reputation System automatically calculates and updates reputation scores for each merchant based on:

- **Credibility-Weighted Ratings** (35% weight): Ratings weighted by reporter credibility
- **Sentiment Analysis** (25% weight): NLP sentiment scores from reports
- **Fraud Risk** (20% weight): Inverse of fraud report ratio
- **Report Volume** (10% weight): Confidence based on number of reports
- **Time Decay** (10% weight): Recent activity boost

## Features

- ✅ Automatic reputation updates on every report submission
- ✅ Credibility-weighted ratings (high-credibility reporters weighted more)
- ✅ Integration with NLP Feature 2 (uses sentiment analysis)
- ✅ Integration with Feature 3 (uses reporter credibility as weight)
- ✅ Time-based weighting (recent reports matter more)
- ✅ Trend analysis (trending_up, trending_down, stable)
- ✅ Fraud risk consideration

## Architecture

```
Report Submission → NLP Analysis → Credibility Update → Reputation Update
```

### Components

1. **ReputationCalculator** (`calculator.py`)
   - Main reputation calculation engine
   - Multi-factor scoring algorithm
   - Credibility-weighted rating calculation
   - Automatic database updates

2. **MerchantReputation** (`models.py`)
   - Pydantic model for reputation data
   - Includes all scoring factors and metrics

3. **Configuration** (`config.py`)
   - Adjustable weights and parameters
   - Credibility weighting settings
   - Time decay settings

## Usage

### Automatic Updates

Reputation is automatically updated when reports are submitted:

```python
from maifds_governance.ingestion.report_handler import ReportHandler

handler = ReportHandler()

# Submit report (reputation automatically updated)
result = handler.submit_report(report_data)
```

### Get Merchant Reputation

```python
# Get current reputation
reputation = handler.get_merchant_reputation("merchant_abc")

print(f"Reputation Score: {reputation['reputation_score']}")
print(f"Average Rating: {reputation['average_rating']}")
print(f"Credibility-Weighted Rating: {reputation['credibility_weighted_rating']}")
print(f"Positive Reports Ratio: {reputation['positive_reports_ratio']}")
print(f"Trend: {reputation['recent_trend']}")
```

### Manual Update

```python
# Manually trigger reputation update
handler.update_merchant_reputation("merchant_abc")
```

### Recalculate All Reputations

```bash
# Recalculate reputation for all merchants
cd src/reputation
python recalculate.py
```

## Scoring Algorithm

### Base Score Calculation

```
reputation = (
    weighted_rating * 0.35 +
    sentiment_score * 0.25 +
    (1 - fraud_risk) * 0.20 +
    volume_score * 0.10 +
    time_decay * 0.10
)
```

### Credibility-Weighted Ratings

Ratings are weighted by reporter credibility:
- Only reporters with credibility ≥ 0.3 are counted
- High-credibility reporters (≥ 0.7) get +0.1 boost
- Formula: `weighted_avg = sum(rating * credibility) / sum(credibility)`

### Factors Explained

1. **Weighted Rating** (0-1)
   - Ratings weighted by reporter credibility
   - Normalized from 1-5 scale to 0-1
   - High-credibility reporters have more influence

2. **Sentiment Score** (0-1)
   - From NLP analysis: positive=1.0, neutral=0.5, negative=0.0
   - Falls back to rating-based sentiment if NLP unavailable

3. **Fraud Risk** (0-1, inverted)
   - Ratio of fraud reports to total reports
   - Applied as penalty (higher fraud = lower reputation)
   - Maximum penalty capped

4. **Volume Score** (0-1)
   - Confidence based on number of reports
   - Optimal around 20 reports
   - Diminishing returns after optimal

5. **Time Decay** (0-1)
   - Exponential decay based on report age
   - Recent reports (within 30 days) weighted more
   - Half-life: 90 days

## Integration with Other Features

### Feature 2: NLP Text Analysis

The reputation system uses NLP sentiment scores:
- Extracts `sentiment` from report metadata
- Converts to numerical scores
- Falls back gracefully if NLP not available

### Feature 3: Reporter Credibility

The reputation system uses reporter credibility as weights:
- Ratings from high-credibility reporters weighted more
- Only credible reporters (≥ 0.3) count in weighted average
- High-credibility reporters (≥ 0.7) get boost

## Database Schema

The `merchants` table stores:

- `merchant_id` (PRIMARY KEY)
- `reputation_score` (0-1, updated automatically)
- `average_rating` (1-5, updated automatically)
- `total_reports` (count)
- `created_at`, `updated_at` (timestamps)

## Example Output

```python
{
    "merchant_id": "merchant_abc",
    "reputation_score": 0.723,
    "total_reports": 25,
    "average_rating": 3.8,
    "credibility_weighted_rating": 4.1,
    "positive_reports_ratio": 0.60,
    "negative_reports_ratio": 0.20,
    "fraud_reports_ratio": 0.08,
    "recent_trend": "trending_up",
    "updated_at": "2025-12-12T01:00:00Z"
}
```

## Configuration

Edit `src/reputation/config.py` to customize:

- **Weights**: Adjust factor weights (must sum to 1.0)
- **Credibility Weighting**: Modify thresholds and boost amounts
- **Time Decay**: Modify recent report window and half-life
- **Volume Scoring**: Adjust optimal report count
- **Fraud Penalty**: Modify penalty amounts

## Testing

```python
# Test reputation calculation
from maifds_governance.reputation.calculator import ReputationCalculator
from maifds_governance.storage.database import DatabaseManager
from maifds_governance.credibility.calculator import CredibilityCalculator

db = DatabaseManager()
cred_calc = CredibilityCalculator(db)
rep_calc = ReputationCalculator(db, cred_calc)

reputation = rep_calc.calculate_reputation("merchant_abc")
print(reputation)
```

## Next Steps

- **Feature 5**: Anti-Gaming & Fraud Detection (will flag suspicious merchants)
- **Feature 6**: Reporting Dashboard (will visualize reputation scores)

