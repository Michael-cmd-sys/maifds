**Test all endpoints (copy/paste)**
**1) Call Triggered Defense**
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{"tx_amount": 5000, "recipient_first_time": 1, "call_to_tx_delta_seconds": 25}'
```
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 5000,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 25,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.7
  }'

```

**2) Click → Tx Correlation (low risk)**
```
curl -X POST http://127.0.0.1:8000/v1/click-tx-correlation \
  -H "Content-Type: application/json" \
  -d '{"tx_amount": 50, "url_reported_flag": 0, "time_between_click_and_tx": 3600}'
```

**3) Click → Tx Correlation (high risk rule trigger)**
```
curl -X POST http://127.0.0.1:8000/v1/click-tx-correlation \
  -H "Content-Type: application/json" \
  -d '{"tx_amount": 7000, "url_reported_flag": 1, "time_between_click_and_tx": 10}'
```

**4) Proactive warning**
```
curl -X POST http://127.0.0.1:8000/v1/proactive-warning \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u001","recent_clicks":3,"recent_calls":2,"new_device_flag":1,"campaign_indicator":1}'
```

**5) Telco webhook notify**
```
curl -X POST http://127.0.0.1:8000/v1/telco-notify \
  -H "Content-Type: application/json" \
  -d '{"incident_id":"inc-001","suspected_number":"+2330000000","affected_accounts":["acc1"],"observed_evidence":{"url_hash":"abc"},"timestamps":{"detected_at":"2025-12-12T10:00:00Z"},"recommended_action":"TEMP_BLOCK"}'
```

**6) Blacklist check (Huawei)**
```
curl -X POST http://127.0.0.1:8000/v1/blacklist/check \
  -H "Content-Type: application/json" \
  -d '{"value":"+233500000000","list_type":"numbers"}'
```

**7) Phishing detector (Huawei)**
```
curl -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \
  -H "Content-Type: application/json" \
  -d '{"url":"http://example.com/login"}'
```
**Health Check**
```
curl http://127.0.0.1:8000/health
```