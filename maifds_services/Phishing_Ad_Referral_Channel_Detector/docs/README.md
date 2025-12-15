# Dataset Information & Usage Guide

## Overview
This document provides information about the datasets used in the **Phishing Ad & Referral Channel Detector** project, including how to obtain them, their structure, and preprocessing steps.

## Dataset Structure

The project uses the following data organization:

```
data/
├── raw/          # Original, immutable datasets
└── processed/    # Cleaned and preprocessed data ready for training
```

### Raw Data Directory (`data/raw/`)
Store your original, unprocessed datasets here. This directory should contain:
- Raw CSV, JSON, or other format files
- Original labels and annotations
- Any supplementary data files

**Important**: Never modify files in this directory. Keep the raw data immutable for reproducibility.

### Processed Data Directory (`data/processed/`)
This directory contains cleaned and preprocessed datasets ready for model training:
- Cleaned and normalized feature data
- Encoded categorical variables
- Train/test/validation splits
- Feature-engineered datasets

## How to Obtain Datasets

### Option 1: Using Existing Datasets
If you have existing datasets for phishing ad and referral channel detection:

1. Place your raw data files in `data/raw/`
2. Run the preprocessing scripts from `src/` to generate processed data
3. Processed data will be saved to `data/processed/`

### Option 2: Collecting New Data
To collect new data for phishing ad and referral channel detection:

1. **Web scraping**: Collect ad data from various web sources
2. **API integration**: Use advertising platform APIs to gather channel data
3. **Manual labeling**: Annotate collected data with phishing/legitimate labels
4. **Data validation**: Verify data quality and consistency

### Option 3: Public Datasets
Consider using publicly available datasets:

- **Phishing websites datasets**: Available on UCI Machine Learning Repository, Kaggle
- **Ad fraud datasets**: Search on academic repositories and research papers
- **URL classification datasets**: Available from various security research organizations

## Data Preprocessing

The preprocessing pipeline (implemented in `src/`) typically includes:

1. **Data cleaning**: Remove duplicates, handle missing values
2. **Feature extraction**: Extract relevant features from URLs, ad content, referral patterns
3. **Feature engineering**: Create derived features for better model performance
4. **Normalization**: Scale numerical features
5. **Encoding**: Convert categorical variables to numerical format
6. **Splitting**: Create train/validation/test sets

## Using the Data

### Training the Model
```bash
cd src/
python train_mindspore.py
```

### Checking Model Performance
```bash
cd src/
python check_model.py
```

### Running the API
```bash
cd src/
python api_mindspore.py
```

## Data Privacy & Security

- Ensure compliance with data protection regulations (GDPR, CCPA, etc.)
- Do not commit sensitive or personal data to version control
- Use `.gitignore` to exclude data directories from Git tracking
- Consider data anonymization for publicly shared datasets

## Dataset Schema

### Expected Features
The model expects datasets with the following types of features:

- **URL-based features**: Domain characteristics, URL length, special characters
- **Content-based features**: Ad text, images, call-to-action elements
- **Behavioral features**: Click patterns, referral chains, user interactions
- **Temporal features**: Time-based patterns, campaign duration

### Labels
- `0`: Legitimate ad/channel
- `1`: Phishing ad/fraudulent channel

## Contributing Data

If you have datasets that could benefit this project:

1. Ensure data is properly anonymized
2. Document the collection methodology
3. Provide clear labeling criteria
4. Include data dictionary/schema

## References

For more information about the project:
- See [QUICKSTART.md](QUICKSTART.md) for quick setup instructions
- See [README_MINDSPORE.md](README_MINDSPORE.md) for MindSpore-specific documentation
- Check the source code in `../src/` for implementation details
