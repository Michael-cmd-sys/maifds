**10 curl tests for click_tx_link_correlation**

 - If your endpoint is POST /v1/click-tx-correlation, use these.

**1) Baseline low risk: old click, low URL risk, small amount**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 60,
    "url_risk_score": 0.1,
    "url_reported_flag": 0,
    "clicked_recently": 0,
    "time_between_click_and_tx": 7200,
    "time_since_last_tx_seconds": 600,
    "tx_hour": 14,
    "tx_dayofweek": 2,
    "url_hash_numeric": 111,
    "device_click_count_1d": 1,
    "user_click_count_1d": 1
  }'
```

**2) Recent click but low URL risk (should stay LOW unless model says otherwise)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 120,
    "url_risk_score": 0.2,
    "url_reported_flag": 0,
    "clicked_recently": 1,
    "time_between_click_and_tx": 40,
    "time_since_last_tx_seconds": 500,
    "tx_hour": 10,
    "tx_dayofweek": 4,
    "url_hash_numeric": 222,
    "device_click_count_1d": 2,
    "user_click_count_1d": 2
  }'
```

**3) Reported URL but click was long ago (rule should NOT fire)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 300,
    "url_risk_score": 0.8,
    "url_reported_flag": 1,
    "clicked_recently": 0,
    "time_between_click_and_tx": 20000,
    "time_since_last_tx_seconds": 2000,
    "tx_hour": 19,
    "tx_dayofweek": 1,
    "url_hash_numeric": 333,
    "device_click_count_1d": 3,
    "user_click_count_1d": 1
  }'
```

**4) High-risk + recent + large amount (rule should fire HIGH)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000,
    "url_risk_score": 0.9,
    "url_reported_flag": 1,
    "clicked_recently": 1,
    "time_between_click_and_tx": 20,
    "time_since_last_tx_seconds": 50,
    "tx_hour": 2,
    "tx_dayofweek": 6,
    "url_hash_numeric": 123456,
    "device_click_count_1d": 10,
    "user_click_count_1d": 3
  }'
```

**5) High-risk + recent + borderline amount (see if rule fires based on median)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 520,
    "url_risk_score": 0.85,
    "url_reported_flag": 0,
    "clicked_recently": 1,
    "time_between_click_and_tx": 60,
    "time_since_last_tx_seconds": 120,
    "tx_hour": 11,
    "tx_dayofweek": 3,
    "url_hash_numeric": 444,
    "device_click_count_1d": 5,
    "user_click_count_1d": 4
  }'
```

**6) Many clicks on device/user but URL risk moderate (model-driven test)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 400,
    "url_risk_score": 0.6,
    "url_reported_flag": 0,
    "clicked_recently": 1,
    "time_between_click_and_tx": 100,
    "time_since_last_tx_seconds": 30,
    "tx_hour": 23,
    "tx_dayofweek": 5,
    "url_hash_numeric": 555,
    "device_click_count_1d": 80,
    "user_click_count_1d": 40
  }'
```

**7) Extremely short click→tx time (stress test)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 350,
    "url_risk_score": 0.7,
    "url_reported_flag": 0,
    "clicked_recently": 0,
    "time_between_click_and_tx": 1,
    "time_since_last_tx_seconds": 10,
    "tx_hour": 9,
    "tx_dayofweek": 0,
    "url_hash_numeric": 666,
    "device_click_count_1d": 4,
    "user_click_count_1d": 2
  }'
```

**8) Valid full payload (your “dashboard-style” payload)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1200,
    "time_since_last_tx_seconds": 30,
    "tx_hour": 2,
    "tx_dayofweek": 6,
    "url_hash_numeric": 123456,
    "url_risk_score": 0.75,
    "url_reported_flag": 0,
    "clicked_recently": 1,
    "time_between_click_and_tx": 50,
    "device_click_count_1d": 10,
    "user_click_count_1d": 3
  }'
```

**9) Zero amount (edge case, should not crash)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 0,
    "url_risk_score": 0.3,
    "url_reported_flag": 0,
    "clicked_recently": 0,
    "time_between_click_and_tx": 1000,
    "time_since_last_tx_seconds": 0,
    "tx_hour": 0,
    "tx_dayofweek": 0,
    "url_hash_numeric": 0,
    "device_click_count_1d": 0,
    "user_click_count_1d": 0
  }'
```

**10) Validation failure test (should 422)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": -5,
    "url_reported_flag": 2,
    "time_between_click_and_tx": -1
  }'
```

**Confirming Fruad Probability**

```bash
curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 250,
    "time_since_last_tx_seconds": 30,
    "tx_hour": 2,
    "tx_dayofweek": 6,
    "url_hash_numeric": 123456,
    "url_risk_score": 0.2,
    "url_reported_flag": 0,
    "clicked_recently": 0,
    "time_between_click_and_tx": 5000,
    "device_click_count_1d": 1,
    "user_click_count_1d": 1
  }'
  ```
