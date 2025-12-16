**✅ 10 curl tests for /v1/proactive-pre-tx-warning**
**1) LOW baseline**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 0,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.1,
  "device_age_days": 365,
  "is_new_device": 0,
  "tx_count_total": 120,
  "avg_tx_amount": 35,
  "max_tx_amount": 150,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 0,
  "user_risk_score": 0.2
}'
```

**2) RULE HIGH (campaign+cohort+high score+recent clicks)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 2,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.8,
  "device_age_days": 5,
  "is_new_device": 1,
  "tx_count_total": 3,
  "avg_tx_amount": 150,
  "max_tx_amount": 600,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.85
}'
```

**3) RULE HIGH (scam calls)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 0,
  "recent_scam_calls_7d": 2,
  "scam_campaign_intensity": 0.7,
  "device_age_days": 120,
  "is_new_device": 0,
  "tx_count_total": 30,
  "avg_tx_amount": 60,
  "max_tx_amount": 300,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.75
}'
```

**4) RULE HIGH (historical fraud)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 0,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.6,
  "device_age_days": 200,
  "is_new_device": 0,
  "tx_count_total": 100,
  "avg_tx_amount": 30,
  "max_tx_amount": 120,
  "historical_fraud_flag": 1,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.7
}'
```

**5) MODEL HIGH (no cohort, but high signals)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 5,
  "recent_scam_calls_7d": 3,
  "scam_campaign_intensity": 0.9,
  "device_age_days": 10,
  "is_new_device": 1,
  "tx_count_total": 5,
  "avg_tx_amount": 120,
  "max_tx_amount": 800,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 0,
  "user_risk_score": 0.9
}'
```

**6) LOW despite high campaign (low user risk)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 1,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.9,
  "device_age_days": 100,
  "is_new_device": 0,
  "tx_count_total": 50,
  "avg_tx_amount": 20,
  "max_tx_amount": 80,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.1
}'
```

**7) Minimal payload (defaults fill in)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "user_risk_score": 0.9,
  "scam_campaign_intensity": 0.9,
  "is_in_campaign_cohort": 1
}'
```

**8) All zeros**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": 0,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.0,
  "device_age_days": 0.0,
  "is_new_device": 0,
  "tx_count_total": 0,
  "avg_tx_amount": 0.0,
  "max_tx_amount": 0.0,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 0,
  "user_risk_score": 0.0
}'
```

**9) Validation error (negative clicks)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning" \
-H "Content-Type: application/json" \
-d '{
  "recent_risky_clicks_7d": -1,
  "user_risk_score": 0.5
}'
```

**10) Validation error (risk score > 1)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning" \
-H "Content-Type: application/json" \
-d '{
  "user_risk_score": 1.2
}'
```