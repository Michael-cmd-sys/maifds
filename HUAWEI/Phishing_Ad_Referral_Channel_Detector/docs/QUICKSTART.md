# 🚀 Quick Start Guide - MindSpore Phishing Detector

Get your phishing detector running in **5 minutes**!

## ⚡ Super Fast Setup

```bash
# 1. Clone/navigate to your project directory
cd HUAWEI/

# 2. Run the setup script (does everything automatically)
chmod +x setup_mindspore.sh
./setup_mindspore.sh

# 3. That's it! The script will:
#    - Create virtual environment
#    - Install MindSpore
#    - Install dependencies
#    - Create config files
#    - Train the model
```

## 🎯 Test It Out

```bash
# Start the API
python api_mindspore.py

# In another terminal, test detection:
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "referrer_url": "http://urgent-verify.xyz/login",
    "ad_id": "TEST_001",
    "ad_text": "URGENT! Your account will be suspended!",
    "landing_domain_age": 5,
    "user_id": "user_123",
    "funnel_path": ["impression", "click", "form"]
  }'
```

**Expected Response:**
```json
{
  "risk_level": "HIGH_RISK_ML",
  "confidence": 0.94,
  "reasons": [
    "Neural network flagged as phishing (confidence: 94%)",
    "Very new domain (age: 5 days)"
  ]
}
```

## 📊 View Statistics

```bash
curl http://localhost:5000/statistics
```

## 🔧 Manual Installation (if script fails)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install MindSpore (choose one)
pip install mindspore==2.2.0              # CPU
pip install mindspore-gpu==2.2.0          # GPU
# For Ascend, visit: https://www.mindspore.cn/install/en

# 3. Install other dependencies
pip install -r requirements_mindspore.txt

# 4. Create directories
mkdir -p data/mindspore_models

# 5. Train model
python train_mindspore.py

# 6. Start API
python api_mindspore.py
```

## 🎮 Interactive Demo

```python
from mindspore_detector import MindSporePhishingDetector

# Initialize
detector = MindSporePhishingDetector()

# Create test signal
test_signal = {
    'referrer_url': 'http://phishing-site.xyz/ad',
    'ad_id': 'AD_PHISH',
    'ad_text': 'URGENT! Click now to claim prize!',
    'landing_domain_age': 3,
    'funnel_path': ['impression', 'click', 'form'],
    'user_id': 'user_test'
}

# Detect
result = detector.detect_phishing(test_signal)
print(f"Risk: {result['risk_level']}")
print(f"Confidence: {result['confidence']:.1%}")
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `mindspore_detector.py` | Main detector class |
| `train_mindspore.py` | Training script |
| `api_mindspore.py` | REST API server |
| `config.json` | Configuration |
| `README_MINDSPORE.md` | Full documentation |

## 🐛 Common Issues

### "Model not trained"
```bash
python train_mindspore.py
```

### "MindSpore not found"
```bash
pip install mindspore==2.2.0
```

### Port 5000 already in use
```bash
# Change port in api_mindspore.py or:
python api_mindspore.py --port 8080
```

## 🎯 What's Detected?

✅ Phishing URLs with high entropy  
✅ New/suspicious domains (< 90 days)  
✅ Urgent/threatening language  
✅ Free prize/money scams  
✅ Account verification tricks  
✅ Suspicious TLDs (.xyz, .tk, .ml)  
✅ Transaction anomalies  

## 📞 API Endpoints

```bash
GET  /health           # Check if API is running
POST /detect           # Detect phishing
POST /batch-detect     # Batch processing
GET  /statistics       # View stats
GET  /model-info       # Model details
POST /train            # Train with new data
GET  /watchlist        # View blacklist
POST /watchlist        # Add domain
```

## 🔥 Production Tips

1. **Use GPU/Ascend** for faster inference
2. **Enable HTTPS** in production
3. **Add API authentication** 
4. **Set up monitoring** (logs, metrics)
5. **Use gunicorn** for production deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_mindspore:app
```

## 💡 Next Steps

1. ✅ Train with your own data
2. ✅ Integrate with your MoMo platform
3. ✅ Set up SMS/webhook notifications
4. ✅ Deploy to production
5. ✅ Monitor and improve

## 📖 Need More Help?

- Full docs: `README_MINDSPORE.md`
- MindSpore: https://www.mindspore.cn/en
- Issues: Check `phishing_detector_mindspore.log`

---

**That's it! You now have an AI-powered phishing detector running with MindSpore! 🎉**