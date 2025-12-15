***1) Safe baseline (expect LOW, no rules)***
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 50,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 3600,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.05
  }'
```

***2) Mild suspicion but not extreme (expect MED maybe, no rules)***
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 250,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 600,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.35
  }'
```

***3) Rule-trigger again (your pattern) (expect HIGH, rule_flag true)***
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

***4) High amount + first-time + not contact but no call proximity (tests rule conditions)***
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 5000,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 5000,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.7
  }'
```

***5) Suspicious NLP only (tests whether NLP alone can raise risk)***
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 100,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 9999,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.95
  }'
```

***6) Edge case: missing/invalid values (expect 422 validation error)***
```
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": -1,
    "recipient_first_time": 2,
    "call_to_tx_delta_seconds": -5,
    "contact_list_flag": 3,
    "nlp_suspicion_score": 2
  }'
  ```
