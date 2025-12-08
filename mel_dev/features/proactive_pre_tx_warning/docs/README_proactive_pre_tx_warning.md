---

# FINAL VERSION — README_proactive_pre_tx_warning.md

```md
# Proactive Pre-Transaction Warning & User Prompting  
### Early Scam-Campaign Detection & Preventive User Protection

This module predicts whether a user should receive a **proactive fraud warning SMS** during scam campaigns.  
Instead of reacting after suspicious activity, the system identifies at-risk users *before* they transact.

---

# 1. What the Feature Does

When a scam wave is detected (phishing blast, call-scam surge, mass URL targeting), this feature:

1. Identifies **vulnerable users** based on behavior, exposure, device signals.
2. Estimates **scam campaign intensity** from URL risk and blacklist data.
3. Computes user-level **risk uplift** (likelihood of being targeted next).
4. Determines whether a proactive warning SMS should be sent.
5. Optionally activates **stricter authentication windows** for high-risk users.

This is a common telco-grade fraud defense strategy used by mobile operators and banks.

---

# 2. Signals Used

## **A. User Behavior (from PaySim)**
- Total transaction count  
- Average transaction size  
- Maximum transaction size  
- Historical fraud flag  

Provides a behavioral baseline for detecting vulnerable or anomalous users.

---

## **B. Device & Call Indicators (from Telco Churn Dataset)**
Used to infer:
- Device age in days  
- New device flag  
- Simulated 7-day scam call count  

New devices and frequent scam calls are strong risk indicators.

---

## **C. URL Exposure (from URL/Phishing Datasets)**
Extracted signals:
- Recent risky clicks (7-day window)  
- URL risk scores  
- Blacklist matches  
- Campaign intensity (global phishing wave measure)

Campaign intensity increases when malicious URL activity becomes widespread.

---

# 3. Feature Engineering Summary

### **User-Level Feature Columns Include:**

- `recent_risky_clicks_7d`  
- `recent_scam_calls_7d`  
- `scam_campaign_intensity`  
- `device_age_days`  
- `is_new_device`  
- `tx_count_total`  
- `avg_tx_amount`  
- `max_tx_amount`  
- `historical_fraud_flag`  
- `is_in_campaign_cohort`  
- `user_risk_score`

The final label is:

should_warn ∈ {0, 1}

yaml
Copy code

Indicating whether the user should receive a proactive SMS.

---

# 4. Model Overview

### **Model Name:** ProactiveWarningModel  
### **Framework:** MindSpore  
### **Type:** Binary classifier (MLP)

#### **Architecture**
Dense(64) → ReLU → Dropout
Dense(32) → ReLU → Dropout
Dense(1) → Sigmoid

yaml
Copy code

#### **Training Details**
- Optimizer: Adam  
- Loss: Binary Cross Entropy  
- Train/Validation: 85/15 split  
- Epochs: 20  
- Output: `proactive_warning_mlp.ckpt`  
- Normalization files: `feature_mean.npy`, `feature_std.npy`

The model predicts **should_warn_probability**.

---

# 5. Rule Engine

High precision rules guarantee user protection even if the model is uncertain.

A user **must** be warned if:
scam_campaign_intensity ≥ 0.4
AND user is in the campaign cohort
AND user_risk_score ≥ 0.6
AND (recent risky clicks OR recent scam calls OR new device OR prior fraud)

yaml
Copy code

The rule is intentionally strict to avoid missing victims during active campaigns.

---

# 6. Ensemble Decision Layer

Final warning decision uses:

### **1. Rule Decision (Overrides Model)**
If the rule fires → immediate warning.

### **2. ML Probability Thresholds**
prob ≥ 0.8 → HIGH RISK (warn)
prob ≥ 0.4 → MEDIUM RISK (warn)
prob < 0.4 → LOW RISK (no warning)

yaml
Copy code

### **Actions Generated**
- `SEND_PROACTIVE_SMS`
- `ENABLE_STRICT_VERIFICATION_WINDOW` (for high-tier risk)
- SMS template with recommended precautions

---

# 7. Folder Structure

proactive_pre_tx_warning/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── docs/
│ ├── README_proactive_pre_tx_warning.md
│ └── data_download_instructions.md
│
├── notebooks/
│ └── exploration.ipynb
│
└── src/
├── config.py
├── data_pipeline.py
├── model.py
├── rules.py
├── inference.py
├── train.py
└── test_inference.py

---

# 8. How to Run

### **Step 1 — Build the training table**

```bash
cd src/
python data_pipeline.py
Step 2 — Train the model
bash
Copy code
python train.py
Step 3 — Run inference
bash
Copy code
python test_inference.py
Produces:

Risk tier

Whether to warn

SMS text

Recommended actions
