# maifds_services Fraud Detection Services - Complete Overview

This document provides an overview of all three fraud detection services.

## 🎯 Three-Layer Fraud Prevention System

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: BLACKLIST                           │
│         Ultra-Fast Lookup (Bloom Filters + Redis)               │
│                                                                  │
│  • 211,531 malicious URLs                                       │
│  • O(1) lookup time (< 1ms)                                     │
│  • Port: 5001                                                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              LAYER 2: ML-BASED DETECTION                        │
│         Phishing Ad & Referral Channel Detector                 │
│                                                                  │
│  • MindSpore neural networks                                    │
│  • Detects phishing ads, URLs, patterns                         │
│  • Port: 5000                                                   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│           LAYER 3: PROACTIVE PREVENTION                         │
│         Campaign Detection & User Warning System                │
│                                                                  │
│  • MindSpore models (LSTM + MLP)                                │
│  • Detects active scam campaigns                                │
│  • Warns vulnerable users proactively                           │
│  • Port: 5002                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Folder Structure

```
maifds_services/
├── Phishing_Ad_Referral_Channel_Detector/     (Port 5000)
│   ├── src/
│   │   ├── train_csv.py
│   │   ├── mindspore_detector.py
│   │   └── api_mindspore.py
│   ├── data/
│   │   ├── raw/dataset.csv (11,055 samples)
│   │   └── processed/models/
│   └── docs/
│
├── Blacklist_Watchlist_Service/               (Port 5001)
│   ├── src/
│   │   ├── blacklist_watchlist_service.py
│   │   ├── manage_blacklist.py
│   │   └── api_blacklist.py
│   ├── data/
│   │   ├── raw/malicious_phish.csv (651k URLs)
│   │   ├── blacklist_db.json (211k URLs)
│   │   └── bloom_filters/
│   └── config_blacklist.json
│
└── Proactive_Warning_Service/                 (Port 5002)
    ├── src/
    │   ├── proactive_warning_service.py
    │   └── api_warning.py
    ├── data/
    │   ├── raw/sample_users.csv
    │   └── models/
    ├── config/
    │   └── warning_config.json
    └── README.md
```

## 🚀 Quick Start - All Services

### Service 1: Phishing Detector (Port 5000)

```bash
cd ~/projet/maifds/maifds_services/Phishing_Ad_Referral_Channel_Detector

# Train (first time)
~/mindspore311_env/bin/python src/train_csv.py

# Run API
~/mindspore311_env/bin/python src/api_mindspore.py
```

**Test:**
```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"referrer_url": "http://urgent-verify.xyz/login", "ad_text": "URGENT! Click now!"}'
```

---

### Service 2: Blacklist (Port 5001)

```bash
cd ~/projet/maifds/maifds_services/Blacklist_Watchlist_Service

# Check stats
~/mindspore311_env/bin/python src/manage_blacklist.py stats

# Run API
~/mindspore311_env/bin/python src/api_blacklist.py --port 5001
```

**Test:**
```bash
curl -X POST http://localhost:5001/check \
  -H "Content-Type: application/json" \
  -d '{"url": "http://br-icloud.com.br"}'
```

---

### Service 3: Proactive Warning (Port 5002)

```bash
cd ~/projet/maifds/maifds_services/Proactive_Warning_Service

# Test
~/mindspore311_env/bin/python src/proactive_warning_service.py

# Run API
~/mindspore311_env/bin/python src/api_warning.py --port 5002
```

**Test:**
```bash
curl -X POST http://localhost:5002/detect-campaign \
  -H "Content-Type: application/json" \
  -d '{"metrics": {"call_volume_last_hour": 150, "pattern_anomaly_score": 0.92, "campaign_type": "call"}}'
```

---

## 🔄 How They Work Together

### Real-World Example: Preventing a Phishing Attack

1. **Blacklist Check (Layer 1)**
   ```
   User clicks suspicious link → Blacklist service checks URL
   → BLOCKED if in database (211k+ known bad URLs)
   ```

2. **ML Detection (Layer 2)**
   ```
   New/unknown URL → Phishing Detector analyzes
   → Uses MindSpore neural network
   → Detects phishing patterns
   → BLOCKED if high risk score
   ```

3. **Proactive Warning (Layer 3)**
   ```
   Multiple users targeted → Campaign detected
   → Identifies vulnerable users (new devices, recent clicks)
   → Sends SMS warnings proactively
   → Enables stricter verification
   ```

### Integration Flow

```python
def process_transaction(transaction):
    # Layer 1: Quick blacklist check
    if blacklist_service.check({'url': transaction.url})['is_blacklisted']:
        return BLOCK_TRANSACTION

    # Layer 2: ML-based phishing detection
    risk = phishing_detector.detect({
        'referrer_url': transaction.url,
        'ad_text': transaction.description
    })
    if risk['risk_level'] == 'HIGH_RISK_ML':
        return BLOCK_TRANSACTION

    # Layer 3: Check if user is in vulnerable cohort
    if proactive_service.is_vulnerable_user(transaction.user_id):
        return REQUIRE_EXTRA_VERIFICATION

    return ALLOW_TRANSACTION
```

## 📊 Service Comparison

| Feature | Phishing Detector | Blacklist | Proactive Warning |
|---------|------------------|-----------|-------------------|
| **Technology** | MindSpore NN | Bloom Filters | MindSpore (LSTM+MLP) |
| **Speed** | ~50ms | < 1ms | ~100ms |
| **Data Size** | 11k samples | 211k URLs | User cohorts |
| **Detection Type** | Real-time | Instant | Predictive |
| **Action** | Block/Flag | Block | Warn/Verify |
| **Port** | 5000 | 5001 | 5002 |

## 🎯 Use Cases

### Phishing Detector (Service 1)
- ✅ New suspicious URL detected
- ✅ Ad content analysis
- ✅ Referral chain analysis
- ✅ Domain age checking

### Blacklist (Service 2)
- ✅ Known bad URLs (211k+)
- ✅ Reported phone numbers
- ✅ Stolen device IDs
- ✅ Historical fraud data

### Proactive Warning (Service 3)
- ✅ Active scam campaigns
- ✅ Mass targeting detection
- ✅ Vulnerable user protection
- ✅ Pre-transaction warnings

## 📈 Performance Metrics

### Phishing Detector
- Training: ~2-5 minutes (11k samples)
- Inference: ~50ms per request
- Accuracy: ~92% (test set)
- Memory: ~500MB

### Blacklist Service
- Lookup: < 1ms (with Redis)
- Database: 211,531 entries
- Bloom filter: 1.14 MB
- False positive rate: 1%

### Proactive Warning
- Campaign detection: ~100ms
- Cohort selection: ~50ms per user
- SMS sending: ~200ms per message
- Models: 3 MindSpore networks

## 🚀 Running All Three Together

### Option 1: Three Terminals

**Terminal 1:**
```bash
cd ~/projet/maifds/maifds_services/Phishing_Ad_Referral_Channel_Detector
~/mindspore311_env/bin/python src/api_mindspore.py
```

**Terminal 2:**
```bash
cd ~/projet/maifds/maifds_services/Blacklist_Watchlist_Service
~/mindspore311_env/bin/python src/api_blacklist.py --port 5001
```

**Terminal 3:**
```bash
cd ~/projet/maifds/maifds_services/Proactive_Warning_Service
~/mindspore311_env/bin/python src/api_warning.py --port 5002
```

### Option 2: Using tmux

```bash
# Start tmux
tmux new -s fraud_detection

# Split into 3 panes
# Ctrl+B then " (split horizontally)
# Ctrl+B then " (split again)

# In each pane, start one service
```

## 🔐 Security & Compliance

### Data Privacy
- User data anonymized with SHA-256 hashes
- No PII stored in logs
- GDPR/CCPA compliant

### SMS Compliance
- Opt-in required
- Frequency limits enforced
- Easy opt-out mechanism

## 📞 API Ports Summary

| Service | Port | Health Check |
|---------|------|--------------|
| Phishing Detector | 5000 | `curl http://localhost:5000/health` |
| Blacklist | 5001 | `curl http://localhost:5001/health` |
| Proactive Warning | 5002 | `curl http://localhost:5002/health` |

## 📚 Documentation

- **Phishing Detector**: `Phishing_Ad_Referral_Channel_Detector/docs/`
- **Blacklist**: `Blacklist_Watchlist_Service/docs/README_BLACKLIST.md`
- **Proactive Warning**: `Proactive_Warning_Service/README.md`
- **Personal Guide**: `PERSONAL_USAGE_GUIDE.md` (not in GitHub)

## 🛠️ Maintenance

### Daily Tasks
- Monitor logs for errors
- Check API health endpoints
- Review detection statistics

### Weekly Tasks
- Review false positive rates
- Update blacklist with new URLs
- Analyze campaign detection accuracy

### Monthly Tasks
- Retrain phishing detector with new data
- Rebuild bloom filters (if many deletions)
- Update ML models with latest patterns

## 💡 Tips & Best Practices

1. **Start with Blacklist**: Fastest, catches known threats
2. **Use ML Detection**: For unknown/new threats
3. **Enable Proactive Warnings**: During high-risk periods
4. **Monitor All Three**: Different services catch different attacks
5. **Keep Data Fresh**: Regular updates improve accuracy

---

**Three services, one goal: Protect users from fraud** 🛡️

**Environment**: `~/mindspore311_env` (Python 3.11.13, MindSpore 2.7.0)
