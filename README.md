# MAIFDS
**AI powered cyber protection for Ghana's MoMo ecosystem**

[![Python 3.9-3.11](https://img.shields.io/badge/python-3.9--3.11-blue.svg)](https://www.python.org/downloads/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.4+-orange.svg)](https://www.mindspore.cn/)

## 🚀 Quick Start

**New to the project?** Get up and running in one command:

```bash
./setup.sh  # Linux/macOS
# or
.\setup.ps1  # Windows
```

⚠️ **Important:** MindSpore requires **Python 3.9, 3.10, or 3.11**. The setup script will automatically try to use a compatible version. If you don't have one installed, see [SETUP.md](SETUP.md) for installation instructions.

That's it! See [SETUP.md](SETUP.md) for detailed instructions.

## 📋 What is MAIFDS?

(Momo AI Fraud Defense System) MAIFDS is a comprehensive fraud Defense and cyber protection system designed for mobile money (MoMo) ecosystems in Africa. It combines multiple AI-powered features to protect users from:

- 📞➡️💸 **Call Triggered Defense (Call → Tx Mitigation)**
- 🎣🛑 **Phishing Ad & Referral Channel Detector**
- 🔗📉 **Click to Transaction Link Correlation & Blocker**
- 🧑‍🤝‍🧑📢 **Customer Reporting & Crowd-Sourced Reputation System**
- ⚡🕵️ **Real-Time Blacklist / Watchlist Service (with Bloom Filters)**
- 🏧🔍 **Agent / Merchant Risk Profiling & Mule Network Defense**
- 🧑‍💼🔔 **Human-in-the-Loop Alerting & Verification Portal**
- 🛑💳 **Proactive Pre-Transaction Warning & User Prompting**
- 📡📨 **Automated Telco Notification & Triage Webhook**
- 📡📨 **Automated User Alert - Via SMS**
- 🧠🔍 **Explainability, Audit Trail & Legal/Privacy Controls**


## 🏗️ Project Structure

```
MAIFDS - Momo AI Fraud Defense System
.
├── API_DOCS
│   └── tests_kit
├── customer_reputation_system
│   ├── config
│   ├── customer_reputation_system_data
│   │   └── data
│   │       └── synthetic
│   ├── data
│   │   ├── database
│   │   ├── processed
│   │   └── raw
│   ├── logs
│   ├── src
│   │   ├── agents
│   │   ├── api
│   │   ├── audit
│   │   ├── correlation
│   │   ├── credibility
│   │   ├── explainability
│   │   ├── infrastructure
│   │   │   └── config
│   │   ├── ingestion
│   │   ├── models
│   │   ├── mule_network
│   │   ├── nlp
│   │   │   └── models
│   │   ├── reputation
│   │   ├── storage
│   │   ├── synthetic_data
│   │   └── utils
│   └── tests
├── maifds_governance
│   ├── audit_service
│   └── privacy
├── maifds_services
│   ├── Blacklist_Watchlist_Service
│   │   ├── data
│   │   │   ├── bloom_filters
│   │   │   ├── processed
│   │   │   └── raw
│   │   ├── docs
│   │   └── src
│   │       └── data
│   │           └── bloom_filters
│   └── Phishing_Ad_Referral_Channel_Detector
│       ├── data
│       │   ├── mindspore_models
│       │   ├── processed
│       │   │   └── models
│       │   └── raw
│       ├── docs
│       └── src
│           └── data
│               └── mindspore_models
├── mel_dev
│   └── features
│       ├── call_triggered_defense
│       │   ├── data
│       │   │   ├── processed
│       │   │   └── raw
│       │   ├── docs
│       │   ├── notebooks
│       │   └── src
│       ├── click_tx_link_correlation
│       │   ├── data
│       │   │   ├── processed
│       │   │   └── raw
│       │   ├── docs
│       │   └── src
│       ├── orchestrator
│       │   ├── data
│       │   │   └── logs
│       │   └── src
│       ├── proactive_pre_tx_warning
│       │   ├── data
│       │   │   ├── processed
│       │   │   └── raw
│       │   ├── docs
│       │   └── src
│       ├── telco_notification_webhook
│       │   ├── data
│       │   │   └── logs
│       │   ├── docs
│       │   └── src
│       └── user_sms_alert
│           ├── data
│           │   └── logs
│           ├── docs
│           └── src
├── rank_0
│   └── om
└── ui
    ├── dist
    ├── public
    │   └── logo
    └── src
        ├── api
        ├── assets
        ├── components
        │   ├── charts
        │   ├── layout
        │   └── ui
        ├── data
        ├── hooks
        ├── layouts
        ├── pages
        ├── theme
        └── utils

109 directories

```

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **uv** (recommended) or pip
- **Git**

### One-Command Setup

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup
./setup.sh
```

### Manual Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# Install all dependencies (centralized)
uv pip install -r requirements.txt

# Install project
uv pip install -e .
```

**📚 Full setup guide:** See [SETUP.md](SETUP.md)

## 📦 Dependencies

All dependencies are centralized in **`requirements.txt`** at the root. This includes:

- **MindSpore** (AI framework) - CPU version by default
- **Data processing**: pandas, numpy, scikit-learn
- **NLP**: transformers, spacy
- **Web/API**: Flask, requests
- **Database**: redis, bitarray
- **And more...**

## 🧠 UI TECH STACK 
- React + TypeScript
- Vite (preferred) or Next.js 
- Tailwind CSS (primary styling) 
- Bootstrap (alerts, modals if useful) 
- JavaScript fetch / axios to connect to Python backend 
- Framer Motion for animations 
- Recharts for charts/graphs 
- Heroicons / Lucide / FontAwesome for icons

See `requirements.txt` for the complete list.

## 🎯 Features

### Customer Reputation System
- ✅ Report ingestion with validation
- ✅ NLP-powered text analysis (sentiment, urgency, credibility)
- ✅ SQLite storage with PostgreSQL migration path
- ✅ Security features (SQL injection, XSS Defense)

### MEL Dev - Fraud Defense
- ✅ **Call Triggered Defense**: MLP + rule-based fraud Defense
- ✅ **Click-TX Link Correlation**: URL risk + transaction timing analysis
- ✅ **Proactive Pre-TX Warning**: Early scam campaign Defense
- ✅ **Telco Notification Webhook**: Incident reporting integration
- ✅ **User Notification Alert - Via SMS**: Incident reporting integration

### maifds_services Services
- ✅ **Phishing Detector**: MindSpore-based phishing Defense
- ✅ **Blacklist Service**: Real-time blacklist management

### maifds_governance
- ✅ **audit_service**
- ✅ **privacy**

## 🚦 Getting Started

### 1. Customer Reputation System

```bash
cd customer-reputation-system
python main.py  # Demo with sample reports
```

### 2. Train NLP Model

```bash
cd customer-reputation-system/src/nlp
python train.py  # Train on reports from database
python test_inference.py  # Test inference
```

### 3. MEL Dev Features

```bash
cd mel_dev/features/call_triggered_defense/src
python train.py  # Train fraud Defense model
python test_inference.py  # Test inference
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific module
pytest customer-reputation-system/tests/

npm install # Build Node
npm run dev # Backend and Front Runs

```



## 📚 Documentation

- **[SETUP.md](SETUP.md)** - Complete setup guide
- **[customer-reputation-system/README.md](customer-reputation-system/README.md)** - Customer reputation system docs
- **[mel_dev/features/](mel_dev/features/)** - Individual feature documentation
- **[maifds_services/SERVICES_OVERVIEW.md](maifds_services/SERVICES_OVERVIEW.md)** - maifds_services services overview
- **[API_DOCS/README.md](API_DOCS/README.md)** - Documentation for Our API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📝 License

A Product built for Huawei Innovation Competition and to help solve the mobile money fraud in Ghana and Africa as a whole.

## 🙏 Acknowledgments

- MindSpore team for the AI framework
- All contributors to the project

## 👥 Development Team
- Sackey Melchizedek Gbine - Leader (https://github.com/Gbine1)
- Cyril Senanu (https://github.com/cysenanu123-oss)
- Michael Awuni (https://github.com/Michael-cmd-sys)

---

**Need help?** Check [SETUP.md](SETUP.md) or open an issue on GitHub.
