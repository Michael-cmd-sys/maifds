# Data Download Instructions  
### Proactive Pre-Transaction Warning & User Prompting Feature

This feature requires several publicly available datasets.  
We combine them to simulate:

- User transaction behavior  
- Device and call-related risk indicators  
- Exposure to phishing URLs  
- Scam campaign intensity

All datasets must be placed in the folder:

mel_dev/features/proactive_pre_tx_warning/data/raw/

yaml
Copy code

---

## 1. PaySim – Mobile Money Transactions  
**Purpose:**  
Baseline user behavior, spending patterns, fraud history.

**Download:**  
https://www.kaggle.com/datasets/ealaxi/paysim1  

**Save as:**  
data/raw/paysim.csv

yaml
Copy code

---

## 2. Telco Customer Churn Dataset  
**Purpose:**  
Used as a proxy for:  
- Device age (“tenure”)  
- New device detection  
- Simulated scam call exposure  

**Download:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn  

**Save as:**  
data/raw/cdr.csv

yaml
Copy code

---

## 3. Malicious vs Benign URL Dataset  
**Purpose:**  
Detect risky URLs, measure phishing exposure, build campaign-intensity signals.

**Download:**  
https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls  

**Save as:**  
data/raw/url_risk.csv

yaml
Copy code

---

## 4. Phishing URL Blacklist (Optional but Recommended)  
**Purpose:**  
Additional source for URL blacklisting and scam-campaign estimation.

**Download:**  
https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset  

**Save as:**  
data/raw/phishing_blacklist.csv

yaml
Copy code

---

## 5. After Downloading  
Your folder structure MUST look like this:

proactive_pre_tx_warning/
└── data/
└── raw/
├── paysim.csv
├── cdr.csv
├── url_risk.csv
└── phishing_blacklist.csv (optional)

yaml
Copy code

---

## 6. Build the Training Table

From the `src` directory:

```bash
cd mel_dev/features/proactive_pre_tx_warning/src
python data_pipeline.py
This generates:

bash
Copy code
data/processed/proactive_warning_training_table.parquet
This file is used to train the Proactive Warning Model.
