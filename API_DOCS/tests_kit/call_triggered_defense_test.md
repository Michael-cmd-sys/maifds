**📌 Call-Triggered Defense — Test Kit**

**Test 1: Safe baseline (trusted contact, long delay)**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 50,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 7200,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.02
  }'
```

- `Expected: LOW, no actions`

**Test 2: First-time recipient, but no call proximity**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 200,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 3600,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.15
  }'
```

- `Expected: LOW`

**Test 3: Suspicious NLP only**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 120,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 9999,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.95
  }'
```

- `Expected: LOW or MEDIUM (model dependent)`

**Test 4: Large amount but trusted contact**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 5000,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 4000,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.1
  }'
```

- `Expected: LOW or MEDIUM`

**Test 5: Near-call scam pattern (classic)**

```bash
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

- `Expected: HIGH, rule triggered`

**Test 6: Near-call but small amount**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 100,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 40,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.4
  }'
```

- `Expected: MEDIUM or LOW`

**Test 7: Large amount, no call, unknown recipient**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 8000,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 8000,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.3
  }'
```

- `Expected: LOW or MEDIUM`

**Test 8: Extremely suspicious NLP + near call**
bash
```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 1500,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 60,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.98
  }'
```

- `Expected: HIGH`

**Test 9: Edge case — zero amount**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 0,
    "recipient_first_time": 0,
    "call_to_tx_delta_seconds": 1000,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.0
  }'
```

- `Expected: LOW`


**Test 10: Validation error (should fail)**

```bash
curl -X POST http://127.0.0.1:8000/v1/call-triggered-defense \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": -100,
    "recipient_first_time": 2,
    "call_to_tx_delta_seconds": -1,
    "contact_list_flag": 3,
    "nlp_suspicion_score": 1.5
  }'
```


- `Expected: HTTP 422`

**Extras (Debug)**
```bash
curl -X POST "http://127.0.0.1:8000/v1/call-triggered-defense?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 250,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 600,
    "contact_list_flag": 1,
    "nlp_suspicion_score": 0.35
  }'


curl -X POST "http://127.0.0.1:8000/v1/call-triggered-defense?debug=true" \
  -H "Content-Type: application/json" \
  -d '{
    "tx_amount": 5000,
    "recipient_first_time": 1,
    "call_to_tx_delta_seconds": 25,
    "contact_list_flag": 0,
    "nlp_suspicion_score": 0.7
  }'
```