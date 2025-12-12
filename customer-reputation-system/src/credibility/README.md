# Feature 3: Reporter Credibility System

A comprehensive system for calculating and maintaining reporter credibility scores based on multiple factors.

## Overview

The Reporter Credibility System automatically calculates and updates credibility scores for each reporter based on:

- **NLP Text Credibility** (40% weight): Average credibility scores from NLP analysis of report text
- **Report Consistency** (25% weight): Consistency of report types and ratings over time
- **Verification Rate** (25% weight): Percentage of verified reports
- **Time Decay** (10% weight): Recent activity boost (recent reports weighted more)

## Features

- ✅ Automatic credibility updates on every report submission
- ✅ Multi-factor scoring algorithm
- ✅ Integration with NLP Feature 2 (uses text credibility scores)
- ✅ Time-based weighting (recent reports matter more)
- ✅ Verification boost for verified reporters
- ✅ Quality metrics consideration

## Architecture

```
Report Submission → NLP Analysis → Credibility Calculation → Database Update
```

### Components

1. **CredibilityCalculator** (`calculator.py`)
   - Main credibility calculation engine
   - Multi-factor scoring algorithm
   - Automatic database updates

2. **ReporterCredibility** (`models.py`)
   - Pydantic model for credibility data
   - Includes all scoring factors and metrics

3. **Configuration** (`config.py`)
   - Adjustable weights and parameters
   - Time decay settings
   - Verification parameters

## Usage

### Automatic Updates

Credibility is automatically updated when reports are submitted:

```python
from src.ingestion.report_handler import ReportHandler

handler = ReportHandler()

# Submit report (credibility automatically updated)
result = handler.submit_report(report_data)
```

### Get Reporter Credibility

```python
# Get current credibility
credibility = handler.get_reporter_credibility("user_123")

print(f"Credibility Score: {credibility['credibility_score']}")
print(f"Total Reports: {credibility['total_reports']}")
print(f"Verified Reports: {credibility['verified_reports']}")
print(f"Average Text Credibility: {credibility['average_text_credibility']}")
print(f"Consistency Score: {credibility['consistency_score']}")
```

### Manual Update

```python
# Manually trigger credibility update
handler.update_reporter_credibility("user_123")
```

### Recalculate All Credibilities

```bash
# Recalculate credibility for all reporters
cd src/credibility
python recalculate.py
```

## Scoring Algorithm

### Base Score Calculation

```
credibility = (
    text_credibility * 0.4 +
    consistency * 0.25 +
    verification_rate * 0.25 +
    time_decay * 0.1
)
```

### Additional Boosts

- **Verification Boost**: +0.2 if reporter has verified reports
- **Quality Boost**: +0.1 * completeness_score (based on report completeness)

### Factors Explained

1. **Text Credibility** (0-1)
   - Average of NLP credibility scores from all reports
   - Falls back to 0.5 if no NLP analysis available

2. **Consistency** (0-1)
   - Based on report type consistency
   - Rating variance consideration
   - Requires minimum 3 reports

3. **Verification Rate** (0-1)
   - Ratio of verified reports to total reports
   - Currently uses `verified_reports` from database

4. **Time Decay** (0-1)
   - Exponential decay based on report age
   - Recent reports (within 30 days) weighted more
   - Half-life: 90 days

## Configuration

Edit `src/credibility/config.py` to customize:

- **Weights**: Adjust factor weights (must sum to 1.0)
- **Time Decay**: Modify recent report window and half-life
- **Consistency**: Change minimum reports and similarity threshold
- **Verification**: Adjust boost amount and minimum requirements

## Database Schema

The `reporters` table stores:

- `reporter_id` (PRIMARY KEY)
- `credibility_score` (0-1, updated automatically)
- `total_reports` (count)
- `verified_reports` (count)
- `created_at`, `updated_at` (timestamps)

## Integration with Other Features

### Feature 2: NLP Text Analysis

The credibility system uses NLP credibility scores from Feature 2:
- Extracts `credibility_score` from report metadata
- Averages across all reports
- Falls back gracefully if NLP not available

### Future Features

- **Feature 4**: Merchant Reputation Scoring (will use reporter credibility as weight)
- **Feature 5**: Anti-Gaming (will flag low-credibility reporters)
- **Feature 6**: Dashboard (will display credibility scores)

## Example Output

```python
{
    "reporter_id": "user_123",
    "credibility_score": 0.782,
    "total_reports": 15,
    "verified_reports": 3,
    "average_text_credibility": 0.75,
    "consistency_score": 0.85,
    "recent_activity_score": 0.92,
    "updated_at": "2025-12-12T01:00:00Z"
}
```

## Testing

```python
# Test credibility calculation
from src.credibility.calculator import CredibilityCalculator
from src.storage.database import DatabaseManager

db = DatabaseManager()
calc = CredibilityCalculator(db)

credibility = calc.calculate_credibility("user_123")
print(credibility)
```

## Next Steps

- **Feature 4**: Merchant Reputation Scoring (will use reporter credibility)
- **Feature 5**: Anti-Gaming & Fraud Detection (will flag suspicious reporters)
- **Feature 6**: Reporting Dashboard (will visualize credibility)

