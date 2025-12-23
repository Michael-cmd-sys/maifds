## 1️⃣ Submit Fraud Report (baseline)

```bash
curl -X POST "http://127.0.0.1:8000/v1/customer-reputation/report/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "reporter_id": "user_001",
    "merchant_id": "merchant_abc",
    "report_type": "fraud",
    "rating": 1,
    "title": "Unauthorized charge",
    "description": "I saw a charge I did not authorize.",
    "transaction_id": "txn_1001",
    "amount": 150.00,
    "metadata": {"platform":"mobile","location":"Accra"}
  }'
  ```

## 2️⃣ Submit Second Report (same merchant)

```bash
curl -X POST "http://127.0.0.1:8000/v1/customer-reputation/report/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "reporter_id": "user_002",
    "merchant_id": "merchant_abc",
    "report_type": "scam",
    "rating": 2,
    "title": "Suspicious payment",
    "description": "Merchant requested payment outside platform.",
    "transaction_id": "txn_1002",
    "amount": 300.00,
    "metadata": {"platform":"web","location":"Kumasi"}
  }'
  ```

## 3️⃣ Get Report by ID

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/report/3505f8ff-9adc-49a7-aa63-ce65578627b1"
```

## 4️⃣ Get CRS System Statistics

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/stats"
```

## 5️⃣ Get Reports for a Merchant

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/merchant/merchant_abc/reports"
```

## 6️⃣ Get Reports by Reporter

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/reporter/user_001/reports"
```

## 7️⃣ Get Agent Risk Score

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/agent/agent_001/risk"
```

## 8️⃣ Get Merchant Risk Assessment

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/merchant/merchant_abc/risk"
```

## 9️⃣ Detect Suspicious Transactions (last 24h)

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/transactions/suspicious?hours=24"
```

## 🔟 Get Risk Alerts (threshold ≥ 0.7)

```bash
curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/alerts?threshold=0.7"
```
