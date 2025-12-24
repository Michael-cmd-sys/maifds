**🚀 MindSpore-Powered Fraud Defense Suite — Feature Branch**
Advanced Mobile Money Fraud Detection & Telco-Integrated Defense Models


**📌 About This Branch**

This branch contains four fully–implemented fraud-detection features, forming the intelligent core of a full-stack telco financial-fraud prevention platform built with MindSpore.

Each feature is modular, production-ready, and exposes:

- a data pipeline  
- a MindSpore model  
- a rule-based expert system  
- an inference engine  
- internal tests and documentation

This branch belongs to Gbine1, team lead and architect of these four modules.

**🧠 Implemented Features (5)**
Feature	Description	ML Model	Rule Engine	Status

| Feature | Description | ML Model | Rule Engine | Status |
|--------|-------------|----------|-------------|--------|
| 📞 Call-Triggered Defense | Detects fraudulent transactions occurring shortly after suspicious calls. | MindSpore MLP | Yes | ✅ Complete |
| 🔗 Click → Transaction Link Correlation | Detects risk based on phishing URL clicks prior to a transaction. | MindSpore MLP | Yes | ✅ Complete |
| ⚠️ Proactive Pre-Transaction Warning | Predicts risk cohorts and proactively warns vulnerable users. | MindSpore MLP | Yes | ✅ Complete |
| 📡 Telco Notification Webhook | Sends structured fraud incidents to telco for investigation and mitigation. | No ML | Webhook & Auditing | ✅ Complete |
| 📡 User Notification (SMS) | Sends structured fraud incidents to User via SMS for investigation and mitigation. | No ML | SMS & Auditing | ✅ Complete |


**🧠 Model Implementations (Feature-Level)**
| Feature | Model Name | Model Type | Framework | Checkpoint / Artifact |
|-------|------------|-----------|-----------|------------------------|
| Call-Triggered Defense | `CallTriggeredDefenseNet` | Tabular MLP Binary Classifier | MindSpore | `call_triggered_defense_mlp.ckpt` |
| Click → Transaction Link Correlation | `ClickTxLinkNet` | Tabular MLP Binary Classifier | MindSpore | `click_tx_link_model.ckpt` |
| Proactive Pre-Transaction Warning | `ProactiveWarningNet` | Tabular MLP Binary Classifier | MindSpore | `proactive_warning_mlp.ckpt` |
| Telco Notification Webhook | — | Event-driven Webhook (No ML) | — | Audit logs (`incidents.jsonl`) |
| User Notification (SMS) | — | Event-driven SMS Alert (No ML) | — | Audit logs (`incidents.jsonl`) |
> All machine-learning models are custom-designed tabular MLPs trained from scratch using MindSpore and combined with deterministic rule engines for high-precision fraud mitigation.

   
**📂 Repository Structure (Branch View)**

```
mel_dev/
├── __pycache__
└── features
    ├── call_triggered_defense
    │   ├── data
    │   │   ├── processed
    │   │   └── raw
    │   ├── docs
    │   ├── notebooks
    │   └── src
    │       └── __pycache__
    ├── click_tx_link_correlation
    │   ├── data
    │   │   ├── processed
    │   │   └── raw
    │   ├── docs
    │   └── src
    │       └── __pycache__
    ├── orchestrator
    │   ├── data
    │   │   └── logs
    │   └── src
    │       └── __pycache__
    ├── proactive_pre_tx_warning
    │   ├── data
    │   │   ├── processed
    │   │   └── raw
    │   ├── docs
    │   └── src
    │       └── __pycache__
    ├── telco_notification_webhook
    │   ├── data
    │   │   └── logs
    │   ├── docs
    │   └── src
    │       └── __pycache__
    └── user_sms_alert
        ├── data
        │   └── logs
        ├── docs
        └── src
            └── __pycache__

42 directories
```


**⚙️ Installation**
**1. Clone repo**
```bash
git clone https://github.com/Michael-cmd-sys/maifds.git
cd maifds_repo
```

**2. Create virtual environment**
```bash
python3 -m venv mindspore_env
source mindspore_env/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. (Optional) Install GPU MindSpore if needed**

Follow official instructions:
https://www.mindspore.cn/install

**🚀 Running Model Pipelines**

Each feature has its own training script:

cd mel_dev/features/<feature_name>/src
python train.py


Each feature also has a test inference script:

python test_inference.py


The webhook tester:

python test_client.py


**🤝 Contribution Guide**

To contribute to this branch:

**1. Checkout a new feature branch**
```bash
git checkout -b feature/<name>
```

**2. Follow existing folder structure**

Each feature must include:
```
data/
docs/
src/
```

**3. Commit cleanly**
```bash
git add .
git commit -m "Describe your change"
git push origin feature/<name>
```

**4. Open Pull Request into `melchizedek_dev`**
Your code will be reviewed by **Gbine1 (Team Lead)**.

**👤 Author**
Gbine1 (Lead Developer & Architect)


**📜 License**
This project is for the Huawei MindSpore Innovation Competition.
Internal academic and research use only.
