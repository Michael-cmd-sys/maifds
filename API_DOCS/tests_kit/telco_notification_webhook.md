## 10 cURL tests for your /v1/telco-notify endpoint

**These hit your FastAPI gateway, which then uses TelcoNotificationClient to POST to your configured TELCO_WEBHOOK_URL.**

**1) Minimal valid payload**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0001",
    "suspected_number": "0559426442",
    "affected_accounts": ["momo:233559426442"],
    "observed_evidence": {"signal":"multiple scam calls"},
    "timestamps": {"detected_at":"2025-12-15T18:10:00Z"},
    "recommended_action": "INVESTIGATE_AND_TEMP_BLOCK"
  }'
```


**2) More evidence fields (multi-feature)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0002",
    "suspected_number": "0244000000",
    "affected_accounts": ["momo:233244000000","momo:233201111111"],
    "observed_evidence": {
      "cdr_ids": ["cdr_hash_1","cdr_hash_2"],
      "click_hashes": ["urlhash_9"],
      "model_scores": {"proactive_pre_tx_warning": 0.98, "call_triggered_defense": 0.91},
      "rule_flags": {"proactive_pre_tx_warning": true}
    },
    "timestamps": {"window_start":"2025-12-15T10:00:00Z","window_end":"2025-12-15T18:00:00Z"},
    "recommended_action": "TEMP_BLOCK_NUMBER"
  }'
```


**3) Empty evidence/timestamps (still valid because dicts default)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0003",
    "suspected_number": "0200000000",
    "affected_accounts": [],
    "observed_evidence": {},
    "timestamps": {},
    "recommended_action": "LOG_ONLY"
  }'
```


**4) Large affected_accounts list**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0004",
    "suspected_number": "0550000000",
    "affected_accounts": ["a1","a2","a3","a4","a5"],
    "observed_evidence": {"note":"burst activity"},
    "timestamps": {"detected_at":"2025-12-15T18:12:00Z"},
    "recommended_action": "INVESTIGATE_CAMPAIGN"
  }'
```


**5) Non-PII “observed_evidence” example**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0005",
    "suspected_number": "0271234567",
    "affected_accounts": ["momo:233271234567"],
    "observed_evidence": {"pattern":"OTP harvest script", "confidence":0.87},
    "timestamps": {"detected_at":"2025-12-15T18:13:00Z"},
    "recommended_action": "ESCALATE_TO_FRAUD_DESK"
  }'
```


**6) Invalid: missing recommended_action (should 422)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-bad-1",
    "suspected_number": "0559426442"
  }'
```


**7) Invalid: empty incident_id (should 422)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "",
    "suspected_number": "0559426442",
    "affected_accounts": [],
    "observed_evidence": {},
    "timestamps": {},
    "recommended_action": "LOG_ONLY"
  }'
```


**8) Wrong type: observed_evidence should be object (dict) (should 422)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-bad-2",
    "suspected_number": "0559426442",
    "affected_accounts": [],
    "observed_evidence": "this-should-be-a-dict",
    "timestamps": {},
    "recommended_action": "LOG_ONLY"
  }'
```


**9) Simulate webhook failure by overriding env (manual)**

**Run your API with:**

- export TELCO_WEBHOOK_URL="http://127.0.0.1:9999/nowhere"


- Then call:
```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0009",
    "suspected_number": "0559426442",
    "affected_accounts": ["momo:233559426442"],
    "observed_evidence": {"test":"force network error"},
    "timestamps": {"detected_at":"2025-12-15T18:20:00Z"},
    "recommended_action": "TEMP_BLOCK_NUMBER"
  }'
```



- Check webhook_errors.jsonl.

**10) High-detail timestamps (good for audits)**

```bash
curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \
  -H "Content-Type: application/json" \
  -d '{
    "incident_id": "inc-0010",
    "suspected_number": "0501112223",
    "affected_accounts": ["momo:233501112223"],
    "observed_evidence": {"rule":"campaign_cohort+recent_calls", "score":0.93},
    "timestamps": {
      "first_seen":"2025-12-15T08:00:00Z",
      "last_seen":"2025-12-15T18:00:00Z",
      "reported_at":"2025-12-15T18:22:00Z"
    },
    "recommended_action": "INVESTIGATE_AND_MONITOR"
  }'
```
