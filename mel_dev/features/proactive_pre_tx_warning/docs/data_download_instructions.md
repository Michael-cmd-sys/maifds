# Data Download & Preparation Instructions  
### Proactive Pre-Transaction Warning & User Prompting Feature

This feature relies on **user-level behavioral signals**, **click/call indicators**, and **campaign-level phishing activity**.  
Because no public dataset directly matches this problem, we combine multiple real datasets and build synthetic signals on top of them.

Below are the required datasets and how to prepare them.

---

## 1. Required Datasets

### **A) PaySim Mobile Money Transactions**
Used for:
- user activity baseline  
- transaction count & spending patterns  
- historical fraud labeling (isFraud)

**Download:**  
https://www.kaggle.com/datasets/ealaxi/paysim1

**Save as:**  
data/raw/paysim.csv

yaml
Copy code

---

### **B) Telco Call Behaviour (Proxy for device age & scam call exposure)**
Used for:
- device_age_days  
- new device identification  
- simulated scam call frequency

**Download:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

**Save as:**  
data/raw/cdr.csv

yaml
Copy code

---

### **C) URL Activity & Phishing Indicators**
Used for:
- URL risk score  
- recent risky clicks  
- campaign intensity estimation

**1. Malicious + Benign URL Dataset**  
https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-urls  

Save as:
data/raw/url_risk.csv

csharp
Copy code

**2. Phishing Blacklist (Optional but recommended)**  
https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset  

Save as:
data/raw/phishing_blacklist.csv

yaml
Copy code

---

## 2. Folder Structure

Place all raw files here:

mel_dev/features/proactive_pre_tx_warning/data/raw/
│
├─ paysim.csv
├─ cdr.csv
├─ url_risk.csv
└─ phishing_blacklist.csv (optional)

yaml
Copy code

---

## 3. Running the Data Pipeline

From within the feature folder:

```bash
cd mel_dev/features/proactive_pre_tx_warning/src
python data_pipeline.py
This generates:

bash
Copy code
data/processed/proactive_warning_training_table.parquet