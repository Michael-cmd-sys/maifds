# Customer Reputation System - Feature 1: Report Ingestion

A robust report submission and ingestion system built with Python and MindSpore for handling customer feedback, complaints, and fraud reports.

## 📋 Overview

This system allows customers to submit reports about merchants, which are validated, processed, and stored in a structured database. It's designed to be the foundation for a crowd-sourced reputation system.

## 🚀 Features

- **Data Validation**: Robust input validation using Pydantic models
- **Security**: Built-in SQL injection and XSS detection
- **Flexible Storage**: SQLite database with easy PostgreSQL migration path
- **Audit Trail**: Raw JSON backups for all reports
- **Type Safety**: Full type hints throughout the codebase
- **Logging**: Comprehensive logging for debugging and monitoring
- **Testing**: Unit tests with pytest

## 📁 Project Structure

```
customer-reputation-system/
├── config/
│   ├── settings.py           # Configuration settings
│   └── logging_config.py     # Logging setup
├── data/
│   ├── raw/                  # Raw JSON backups
│   ├── processed/            # Processed data
│   └── database/             # SQLite database
├── src/
│   ├── models/
│   │   └── report_model.py   # Pydantic data models
│   ├── ingestion/
│   │   └── report_handler.py # Main ingestion logic
│   └── storage/
│       ├── schemas.py        # Database schemas
│       └── database.py       # Database operations
├── tests/
│   └── test_models.py        # Unit tests
├── main.py                   # Demo script
└── requirements.txt          # Dependencies
```

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd customer-reputation-system
```

2. **Create virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Running the Demo

```bash
python main.py
```

This will:

- Initialize the database
- Submit 5 sample reports
- Display statistics
- Demonstrate report retrieval

### Submitting a Report

```python
from src.ingestion.report_handler import ReportHandler

# Initialize handler
handler = ReportHandler()

# Create report data
report_data = {
    "reporter_id": "user_123",
    "merchant_id": "merchant_abc",
    "report_type": "fraud",
    "rating": 1,
    "title": "Unauthorized charge",
    "description": "I was charged twice for the same transaction without authorization.",
    "transaction_id": "txn_456",
    "amount": 150.00,
    "metadata": {
        "platform": "mobile",
        "location": "New York, USA"
    }
}

# Submit report
result = handler.submit_report(report_data)
print(result)
```

### Retrieving Reports

```python
# Get a specific report
report = handler.get_report("report-id-here")

# Get all reports for a merchant
merchant_reports = handler.get_merchant_reports("merchant_abc")

# Get all reports by a reporter
reporter_reports = handler.get_reporter_reports("user_123")

# Get system statistics
stats = handler.get_statistics()
```

## 📊 Report Data Model

### Required Fields

- `reporter_id`: User who submitted the report
- `merchant_id`: Merchant being reported
- `report_type`: One of `fraud`, `service_issue`, `technical`, `other`
- `title`: Brief description (3-200 characters)
- `description`: Detailed description (10-5000 characters)

### Optional Fields

- `rating`: Integer 1-5
- `transaction_id`: Associated transaction
- `amount`: Transaction amount
- `metadata`: Platform, location, device info

### Auto-Generated Fields

- `report_id`: Unique UUID
- `timestamp`: UTC timestamp

## 🔒 Security Features

- **Input Validation**: All fields validated with Pydantic
- **SQL Injection Protection**: Pattern detection in text fields
- **XSS Protection**: HTML/JavaScript pattern detection
- **ID Format Validation**: Alphanumeric, hyphens, underscores only
- **Text Sanitization**: Automatic whitespace normalization

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📝 Database Schema

### Reports Table

- `report_id` (PRIMARY KEY)
- `timestamp`
- `reporter_id`
- `merchant_id`
- `report_type`
- `rating`
- `title`
- `description`
- `transaction_id`
- `amount`
- `metadata_json`
- `created_at`

### Reporters Table

- `reporter_id` (PRIMARY KEY)
- `credibility_score`
- `total_reports`
- `verified_reports`
- `created_at`
- `updated_at`

### Merchants Table

- `merchant_id` (PRIMARY KEY)
- `merchant_name`
- `total_reports`
- `average_rating`
- `reputation_score`
- `created_at`
- `updated_at`

## 🔧 Configuration

Edit `config/settings.py` to customize:

- Database path
- Data directories
- Validation rules
- Text length constraints
- Logging settings

## 📈 Next Steps

This is Feature 1 of the Customer Reputation System. Upcoming features:

- **Feature 2**: Text Analysis & Classification (NLP with MindSpore)
- **Feature 3**: Reporter Credibility System
- **Feature 4**: Merchant Reputation Scoring
- **Feature 5**: Anti-Gaming & Fraud Detection
- **Feature 6**: Reporting Dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
