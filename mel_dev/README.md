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

**🧠 Implemented Features (4 / 10)**
Feature	Description	ML Model	Rule Engine	Status

| Feature | Description | ML Model | Rule Engine | Status |
|--------|-------------|----------|-------------|--------|
| 📞 Call-Triggered Defense | Detects fraudulent transactions occurring shortly after suspicious calls. | MindSpore MLP | Yes | ✅ Complete |
| 🔗 Click → Transaction Link Correlation | Detects risk based on phishing URL clicks prior to a transaction. | MindSpore MLP | Yes | ✅ Complete |
| ⚠️ Proactive Pre-Transaction Warning | Predicts risk cohorts and proactively warns vulnerable users. | MindSpore MLP | Yes | ✅ Complete |
| 📡 Telco Notification Webhook | Sends structured fraud incidents to telco for investigation and mitigation. | No ML | Webhook & Auditing | ✅ Complete |


**🧠 Model Implementations (Feature-Level)**
| Feature | Model Name | Model Type | Framework | Checkpoint / Artifact |
|-------|------------|-----------|-----------|------------------------|
| Call-Triggered Defense | `CallTriggeredDefenseNet` | Tabular MLP Binary Classifier | MindSpore | `call_triggered_defense_mlp.ckpt` |
| Click → Transaction Link Correlation | `ClickTxLinkNet` | Tabular MLP Binary Classifier | MindSpore | `click_tx_link_model.ckpt` |
| Proactive Pre-Transaction Warning | `ProactiveWarningNet` | Tabular MLP Binary Classifier | MindSpore | `proactive_warning_mlp.ckpt` |
| Telco Notification Webhook | — | Event-driven Webhook (No ML) | — | Audit logs (`incidents.jsonl`) |
> All machine-learning models are custom-designed tabular MLPs trained from scratch using MindSpore and combined with deterministic rule engines for high-precision fraud mitigation.

   
**📂 Repository Structure (Branch View)**

```
mel_dev/
│
├── features/
│   ├── call_triggered_defense/
│   │   ├── data/
│   │   │   ├── raw/                # (ignored in Git)
│   │   │   └── processed/
│   │   │       └── call_tx_training_table.parquet
│   │   ├── docs/
│   │   │   ├── DATA_SETS.md
│   │   │   └── README_call_triggered_defense.md
│   │   ├── notebooks/
│   │   └── src/
│   │       ├── config.py
│   │       ├── data_pipeline.py
│   │       ├── model.py
│   │       ├── rules.py
│   │       ├── inference.py
│   │       └── train.py
│
│   ├── click_tx_link_correlation/
│   │   ├── data/
│   │   ├── docs/
│   │   ├── notebooks/
│   │   └── src/
│
│   ├── proactive_pre_tx_warning/
│   │   ├── data/
│   │   ├── docs/
│   │   ├── notebooks/
│   │   └── src/
│
│   └── telco_notification_webhook/
│       ├── data/
│       │   └── logs/
│       ├── docs/
│       └── src/
│           ├── client.py
│           ├── schemas.py
│           ├── storage.py
│           └── test_client.py
│
└── requirements.txt
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
