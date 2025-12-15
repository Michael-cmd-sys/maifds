# HUAWEI Fraud Detection Services

This repository contains three integrated fraud detection services built with MindSpore for mobile money platforms.

## Services

1. **Phishing_Ad_Referral_Channel_Detector** - Detects phishing attempts and malicious ad referrals
2. **Blacklist_Watchlist_Service** - Manages blacklists and watchlists using Bloom filters
3. **Proactive_Warning_Service** - Proactively warns users about detected scam campaigns

## Data Structure

### What's Included in This Repository

The repository includes **processed data and trained models** that are ready to use:

- **Model Checkpoints** (`.ckpt` files): Pre-trained MindSpore models
  - `Phishing_Ad_Referral_Channel_Detector/data/mindspore_models/*.ckpt` - 5 phishing detection model checkpoints
  - `Phishing_Ad_Referral_Channel_Detector/data/processed/models/*.ckpt` - 10 processed model checkpoints

- **Model Artifacts**:
  - `*.pkl` files - Scalers and feature configurations for the models

- **Bloom Filters** (3.5 MB total):
  - `Blacklist_Watchlist_Service/data/bloom_filters/*.bloom` - Efficient blacklist data structures

### What's NOT Included (Raw Data)

To keep the repository size manageable, **raw datasets are excluded** from version control:

- `*/data/raw/*.csv` - Large raw CSV datasets (144MB+)
- `*/data/blacklist_db.json` - Large blacklist database (53MB)

### How to Obtain Raw Data

If you need the raw datasets for training or testing:

1. **Phishing Detection Dataset**
   - File: `Phishing_Ad_Referral_Channel_Detector/data/raw/dataset.csv`
   - Source: [Provide dataset source or download link]
   - Size: ~844 KB

2. **Fraud Transaction Dataset (Credit Card)**
   - File: `Proactive_Warning_Service/data/raw/creditcard.csv`
   - Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) or similar
   - Size: ~144 MB
   - Note: Place in `Proactive_Warning_Service/data/raw/`

3. **Malicious URLs/Phishing Data**
   - Files:
     - `Blacklist_Watchlist_Service/data/raw/malicious_phish.csv` (44 MB)
     - `Blacklist_Watchlist_Service/data/raw/blacklist_malicious_urls.csv` (25 MB)
   - Source: [Provide malicious URL dataset source]
   - Note: Place in `Blacklist_Watchlist_Service/data/raw/`

4. **Blacklist Database**
   - File: `Blacklist_Watchlist_Service/data/blacklist_db.json`
   - This file is generated from the raw data above
   - Run the blacklist service to regenerate it from raw CSV files

## Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # For MindSpore:
   pip install -r requirements_mindspore.txt
   ```

3. (Optional) Download raw datasets if you need to retrain models

4. Run individual services - see each service's README:
   - [Phishing_Ad_Referral_Channel_Detector/docs/README.md](Phishing_Ad_Referral_Channel_Detector/docs/README.md)
   - [Proactive_Warning_Service/README.md](Proactive_Warning_Service/README.md)
   - Blacklist_Watchlist_Service/docs/README.md (if exists)

## Configuration

Main configuration file: `config.json`

Each service has its own configuration in its respective `config/` directory.

## System Requirements

- Python 3.11+
- MindSpore 2.x
- See `requirements_mindspore.txt` for full dependencies

## Notes

- The `.gitignore` is configured to exclude large raw data files but include trained models
- Model checkpoints are included for immediate use without needing to retrain
- If you need to rebuild the blacklist database from scratch, place raw CSV files in `Blacklist_Watchlist_Service/data/raw/` and run the service

## Contributing

When contributing:
- Do NOT commit large raw datasets (*.csv files > 10MB)
- Do NOT commit the blacklist_db.json file
- DO commit trained model checkpoints if they improve performance
- DO update this README if you add new data sources

## License

[Add your license information here]
