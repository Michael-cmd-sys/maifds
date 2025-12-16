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

MAIFDS is a comprehensive fraud detection and cyber protection system designed for mobile money (MoMo) ecosystems. It combines multiple AI-powered features to protect users from:

- 📞➡️💸 **Call Triggered Defense (Call → Tx Mitigation)**
- 🎣🛑 **Phishing Ad & Referral Channel Detector**
- 🔗📉 **Click to Transaction Link Correlation & Blocker**
- 🧑‍🤝‍🧑📢 **Customer Reporting & Crowd-Sourced Reputation System**
- ⚡🕵️ **Real-Time Blacklist / Watchlist Service (with Bloom Filters)**
- 🏧🔍 **Agent / Merchant Risk Profiling & Mule Network Detection**
- 🧑‍💼🔔 **Human-in-the-Loop Alerting & Verification Portal**
- 🛑💳 **Proactive Pre-Transaction Warning & User Prompting**
- 📡📨 **Automated Telco Notification & Triage Webhook**
- 🧠🔍 **Explainability, Audit Trail & Legal/Privacy Controls**


## 🏗️ Project Structure

```
maifds/
├── customer-reputation-system/    # Report ingestion & NLP analysis
│   ├── Feature 1: Report Ingestion ✅
│   └── Feature 2: NLP Text Analysis ✅
│
├── mel_dev/                        # Production fraud detection features
│   └── features/
│       ├── call_triggered_defense/
│       ├── click_tx_link_correlation/
│       ├── proactive_pre_tx_warning/
│       └── telco_notification_webhook/
│
└── maifds_services/                         # Enterprise services
    ├── Phishing_Ad_Referral_Channel_Detector/
    ├── Blacklist_Watchlist_Service/
    └── Proactive_Warning_Service/
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

See `requirements.txt` for the complete list.

## 🎯 Features

### Customer Reputation System
- ✅ Report ingestion with validation
- ✅ NLP-powered text analysis (sentiment, urgency, credibility)
- ✅ SQLite storage with PostgreSQL migration path
- ✅ Security features (SQL injection, XSS detection)

### MEL Dev - Fraud Detection
- ✅ **Call Triggered Defense**: MLP + rule-based fraud detection
- ✅ **Click-TX Link Correlation**: URL risk + transaction timing analysis
- ✅ **Proactive Pre-TX Warning**: Early scam campaign detection
- ✅ **Telco Notification Webhook**: Incident reporting integration

### maifds_services Services
- ✅ **Phishing Detector**: MindSpore-based phishing detection
- ✅ **Blacklist Service**: Real-time blacklist management
- ✅ **Proactive Warning**: User protection services

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
python train.py  # Train fraud detection model
python test_inference.py  # Test inference
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Test specific module
pytest customer-reputation-system/tests/
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
- Sackey Melchizedek Gbine (Leader)
- Cyril Senanu (https://github.com/cysenanu123-oss)
- Michael Awuni (https://github.com/Michael-cmd-sys)

---

**Need help?** Check [SETUP.md](SETUP.md) or open an issue on GitHub.
