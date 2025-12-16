## FastAPI endpoint
## POST /v1/user-sms-alert

- **Example (local Ghana number):**

```bash
curl -X POST "http://127.0.0.1:8000/v1/user-sms-alert" \
  -H "Content-Type: application/json" \
  -d '{
    "phone": "0559426442",
    "message": "MAIFDS ALERT: Suspicious activity detected. Do NOT share OTP/PIN.",
    "ref": "sms-test-001"
  }'
```

- ## Example (+233 format):

```bash
curl -X POST "http://127.0.0.1:8000/v1/user-sms-alert" \
  -H "Content-Type: application/json" \
  -d '{
    "phone": "+233559426442",
    "message": "MAIFDS ALERT: Fraud risk detected. Verify before sending money.",
    "ref": "sms-test-002"
  }'
```


- ## Expected response

- ## On success you should see something like:

```bash
{
  "feature": "user_sms_alert",
  "result": {
    "recipient_normalized": "233559426442",
    "sms_text": "MAIFDS ALERT: ...",
    "provider": {
      "status_code": 200,
      "success": true,
      "provider_code": "SMS01",
      "provider_message": "Success",
      "raw_body": "{\"status\":1,\"code\":\"SMS01\",\"message\":\"Success\",\"data\":null,\"go\":null}",
      "request_id": "sms-test-001"
    }
  }
}
```

- `Notes / best practices`

- `Always include a ref (especially from the orchestrator). It helps trace incidents + dashboard reporting.`

- `For production, secure the endpoint (auth, rate limiting) to avoid abuse.`

- `Consider adding message throttling per phone number (to prevent spamming a user).`



## 10 curl tests for `/v1/orchestrate`

### 1) LOW risk (no actions)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-low-001",
  "incident_id":"inc-low-001",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":0,
    "recent_scam_calls_7d":0,
    "scam_campaign_intensity":0.1,
    "device_age_days":365,
    "is_new_device":0,
    "tx_count_total":120,
    "avg_tx_amount":35,
    "max_tx_amount":150,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":0,
    "user_risk_score":0.2
  }
}'
```

## 2) MEDIUM-ish (expect SMS only if your policy calls it MEDIUM; if model outputs LOW then none)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-med-002",
  "incident_id":"inc-med-002",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":1,
    "recent_scam_calls_7d":1,
    "scam_campaign_intensity":0.55,
    "device_age_days":90,
    "is_new_device":0,
    "tx_count_total":25,
    "avg_tx_amount":70,
    "max_tx_amount":400,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.6
  }
}'
```

## 3) HIGH via proactive rule (SMS + Telco)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-high-003",
  "incident_id":"inc-high-003",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "affected_accounts":["momo:233559426442"],
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":2,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":5,
    "is_new_device":1,
    "tx_count_total":3,
    "avg_tx_amount":150,
    "max_tx_amount":600,
    "historical_fraud_flag":1,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.85
  }
}'
```

## 4) HIGH via call-triggered defense only (SMS + Telco)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-call-004",
  "incident_id":"inc-call-004",
  "user_phone":"0559426442",
  "suspected_number":"0200000000",
  "call_triggered_defense":{
    "tx_amount":5000,
    "recipient_first_time":1,
    "call_to_tx_delta_seconds":25,
    "contact_list_flag":0,
    "nlp_suspicion_score":0.75
  }
}'
```

## 5) HIGH via click-tx correlation only (SMS + Telco)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-click-005",
  "incident_id":"inc-click-005",
  "user_phone":"0559426442",
  "suspected_number":"0271234567",
  "click_tx_correlation":{
    "tx_amount":1200,
    "url_reported_flag":1,
    "time_between_click_and_tx":15,
    "url_risk_score":0.95,
    "clicked_recently":1,
    "device_click_count_1d":6,
    "user_click_count_1d":4
  }
}'
```

## 6) Full-stack HIGH (all three features) + debug

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-all-006",
  "incident_id":"inc-all-006",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "affected_accounts":["momo:233559426442"],
  "call_triggered_defense":{
    "tx_amount":3000,
    "recipient_first_time":1,
    "call_to_tx_delta_seconds":30,
    "contact_list_flag":0,
    "nlp_suspicion_score":0.8
  },
  "click_tx_correlation":{
    "tx_amount":1200,
    "url_reported_flag":1,
    "time_between_click_and_tx":20,
    "url_risk_score":0.9,
    "clicked_recently":1,
    "device_click_count_1d":5,
    "user_click_count_1d":3
  },
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":4,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":3,
    "is_new_device":1,
    "tx_count_total":4,
    "avg_tx_amount":120,
    "max_tx_amount":900,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.9
  }
}'
```

## 7) User phone in +233 format (should still send SMS)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-plus-007",
  "incident_id":"inc-plus-007",
  "user_phone":"+233559426442",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":2,
    "recent_scam_calls_7d":1,
    "scam_campaign_intensity":0.85,
    "device_age_days":7,
    "is_new_device":1,
    "tx_count_total":5,
    "avg_tx_amount":130,
    "max_tx_amount":700,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.8
  }
}'
```

## 8) HIGH but missing user_phone (should do Telco only)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-telco-only-008",
  "incident_id":"inc-telco-only-008",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":3,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":2,
    "is_new_device":1,
    "tx_count_total":2,
    "avg_tx_amount":200,
    "max_tx_amount":900,
    "historical_fraud_flag":1,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.92
  }
}'
```

## 9) HIGH but missing suspected_number (should do SMS only)

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-sms-only-009",
  "incident_id":"inc-sms-only-009",
  "user_phone":"0559426442",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":3,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":2,
    "is_new_device":1,
    "tx_count_total":2,
    "avg_tx_amount":200,
    "max_tx_amount":900,
    "historical_fraud_flag":1,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.92
  }
}'
```

## 10) Bad request (negative value) — should return 422 validation error

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-bad-010",
  "incident_id":"inc-bad-010",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":-1,
    "user_risk_score":0.5
  }
}'
```



## Retest curl (HIGH risk → should send BOTH)

- `This includes ref + incident_id:`

```bash
curl -X POST "http://127.0.0.1:8000/v1/orchestrate?debug=true" \
-H "Content-Type: application/json" \
-d '{
  "ref":"orch-0001",
  "incident_id":"inc-0001",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "affected_accounts":["momo:233559426442"],
  "call_triggered_defense":{
    "tx_amount":3000,
    "recipient_first_time":1,
    "call_to_tx_delta_seconds":30,
    "contact_list_flag":0,
    "nlp_suspicion_score":0.8
  },
  "click_tx_correlation":{
    "tx_amount":1200,
    "url_reported_flag":1,
    "time_between_click_and_tx":20,
    "url_risk_score":0.9,
    "clicked_recently":1,
    "device_click_count_1d":5,
    "user_click_count_1d":3
  },
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":4,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":3,
    "is_new_device":1,
    "tx_count_total":4,
    "avg_tx_amount":120,
    "max_tx_amount":900,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.9
  }
}'
``` 