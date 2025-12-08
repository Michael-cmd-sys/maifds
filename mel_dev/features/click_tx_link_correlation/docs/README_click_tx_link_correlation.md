README – Click → Transaction Link Correlation & Blocker

click_tx_link_correlation/
│
├── data/
│   ├── raw/
│   │   ├── transactions.csv
│   │   ├── url_risk.csv
│   │   └── phishing_blacklist.csv (optional)
│   └── processed/
│       └── click_tx_training_table.parquet
│
├── docs/
│   ├── README_click_tx_link_correlation.md
│   └── data_download_instructions.md
│
├── notebooks/
│   └── exploration.ipynb
│
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── model.py
    ├── train.py
    ├── rules.py
    ├── inference.py
    └── test_inference.py

1. Overview

This feature detects fraudulent transactions that occur shortly after a user clicks a suspicious link.
It correlates:

User click history

URL phishing risk scores

Transaction timing

Device/session behaviour

to determine whether an outgoing financial transaction should be:

Allowed

Held for OTP verification

Blocked as high-risk

Escalated to telco/ops

This feature follows the same architecture as the Call Triggered Defense engine:
Data Pipeline → ML Model → Rule Engine → Ensemble → Risk Output & Actions.

2. Core Logic (How Detection Works)
2.1 Click → Transaction Correlation

For each transaction, the system simulates or reads:

The last URL the user clicked

How long before the transaction the click occurred

Whether the link was malicious or in a blacklist

Device/user click frequency (behavioural indicator)

Key feature:

time_between_click_and_tx = tx_timestamp – click_timestamp


If this number is small, the risk sharply increases.

2.2 URL Risk Scoring

From url_risk.csv + phishing_blacklist.csv, we generate:

url_hash_numeric → SHA-256 → 64-bit integer

url_risk_score → 0 (benign) to 1 (phishing/malicious)

url_reported_flag → 1 if URL appears in known phishing lists

The pipeline automatically detects the URL column, even if it is named differently in the dataset.

2.3 Rule Engine (High Precision)

A transaction is classified HIGH RISK if all conditions are met:

clicked_recently == 1
AND (url_risk_score >= 0.7 OR url_reported_flag == 1)
AND amount >= 2 × median_transaction_amount


This gives the classic "don’t negotiate with fraud" behavior:

If user clicked a malicious link within 5 minutes

AND is now sending a large payment
→ Block or hold the transaction regardless of ML score.

2.4 Machine Learning Model (MindSpore MLP)

Model: ClickTxLinkModel
Framework: MindSpore
Input: 11 engineered features
Architecture:

Dense → ReLU → Dropout

Dense → ReLU → Dropout

Dense → Sigmoid (fraud probability)

Output: fraud_probability ∈ [0, 1]

The model learns patterns such as:

Users who click dangerous links tend to transact quickly

High URL risk correlates with high fraud

Abnormal behaviour in user click frequency

2.5 Ensemble Logic

The final risk outcome is:

If rule_flag == True → HIGH RISK
Else if fraud_probability ≥ 0.8 → HIGH RISK
Else if fraud_probability ≥ 0.4 → MEDIUM RISK
Else → LOW RISK


Recommended actions:

Risk Level	Actions
HIGH	High-risk hold, SMS warning, notify telco ops
MEDIUM	SMS warning
LOW	Allow transaction
3. Dataset Requirements & Download Instructions

Place all raw CSV files under:

mel_dev/features/click_tx_link_correlation/data/raw/

3.1 Required Datasets
A) Transactions Dataset

You can use any e-commerce / financial transaction dataset.
Expected columns (auto-detected):

timestamp or tx_timestamp

amount

user_id (optional)

Rename your file as:

transactions.csv

B) URL Risk Dataset (Malicious vs Benign)

Kaggle (verified):
https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls

Rename downloaded file to:

url_risk.csv


Dataset is used to generate:

URL hash

URL risk score

Domain-level phishing risk

C) Phishing Blacklist (Optional)

Kaggle (verified):
https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset

Rename to:

phishing_blacklist.csv


Used for:

Hard blacklisting

Forcing rule_flag = True for known dangerous links

4. Feature Folder Structure


5. How to Run
Step 1 — Build Training Table
cd mel_dev/features/click_tx_link_correlation/src
python data_pipeline.py


Generates:

data/processed/click_tx_training_table.parquet

Step 2 — Train Model
python train.py


Saves:

click_tx_link_model.ckpt

Step 3 — Test Inference
python test_inference.py


Example output:

{
  'fraud_probability': 0.03,
  'rule_flag': True,
  'risk_level': 'HIGH',
  'reason': 'Recent click on high-risk / reported URL with large transaction.',
  'actions': ['HIGH_RISK_HOLD', 'SMS_USER_WARNING', 'NOTIFY_TELCO_OPS']
}

6. Key Files Explained
File	Purpose
data_pipeline.py	Loads raw datasets, extracts URL & click behaviour, builds full training table
model.py	MindSpore MLP architecture for fraud scoring
train.py	Training loop, normalization, checkpoint saving
rules.py	High-precision rule logic
inference.py	Ensemble of rule + ML → risk level
test_inference.py	Sample inference script
7. Why This Feature Is Powerful

Real phishing attacks often begin with a malicious link → then a transaction.

Combining timing, URL risk, and behavioural features drastically boosts detection accuracy.

Rule engine guarantees high precision on dangerous URLs.

ML model captures subtle behaviour patterns.

Ensemble balances precision + recall.

This creates a telecom-grade defense layer that is suitable for banks, mobile money providers, and telcos.