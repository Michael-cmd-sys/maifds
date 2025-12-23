## 10 Audit CURL tests (ALL same request shape)

- `These are audit ingestion tests. Not prediction.`

- `Copy one-by-one exactly (don’t paste multiple at once).`

## 1) user_login (failed) HIGH

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "user_login",
      "priority": "high",
      "source": "api",
      "user_id": "user_101",
      "data": { "action": "login_attempt", "outcome": "failed" },
      "metadata": { "ip": "197.0.0.1", "device": "android" }
    }
  }'
  ```

## 2) user_login (success) MEDIUM

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "user_login",
      "priority": "medium",
      "source": "web",
      "user_id": "user_101",
      "data": { "outcome": "success" }
    }
  }'
  ```

## 3) data_access (read transactions)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "data_access",
      "priority": "medium",
      "source": "backend",
      "user_id": "analyst_01",
      "data": { "table": "transactions", "operation": "read" }
    }
  }'
  ```

## 4) model_prediction (CRS score logged)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "model_prediction",
      "priority": "low",
      "source": "customer_reputation_system",
      "data": { "model": "CRS", "score": 0.87, "entity_type": "merchant", "entity_id": "merchant_001" }
    }
  }'
  ```

## 5) config_change (risk threshold)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "config_change",
      "priority": "critical",
      "source": "admin_panel",
      "user_id": "admin_01",
      "data": { "parameter": "risk_threshold", "old": 0.80, "new": 0.60 }
    }
  }'
  ```

## 6) system_error (service timeout)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "system_error",
      "priority": "high",
      "source": "orchestrator",
      "data": { "error": "timeout", "service": "blacklist_watchlist" }
    }
  }'
  ```

## 7) access_denied (RBAC/privacy style)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "access_denied",
      "priority": "high",
      "source": "privacy_access_control",
      "user_id": "agent_007",
      "data": { "resource": "customer_pii", "purpose": "fraud_review", "reason": "insufficient_role" }
    }
  }'
  ```

## 8) anomaly_detected

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "anomaly_detected",
      "priority": "high",
      "source": "correlation_engine",
      "data": { "pattern": "rapid_failed_logins", "count": 12, "window_minutes": 5 }
    }
  }'
  ```

## 9) consent_change

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \
  -H "Content-Type: application/json" \
  -d '{
    "method": "ingest_event",
    "payload": {
      "event_type": "consent_change",
      "priority": "low",
      "source": "mobile_app",
      "user_id": "user_999",
      "data": { "consent": "marketing", "status": "revoked" }
    }
  }'
  ```

## 10) health check

```bash
curl -s "http://127.0.0.1:8000/v1/governance/audit/health"
```

## Test with CURL
- `Stats (after sending events)`
```bash 
curl -s "http://127.0.0.1:8000/v1/governance/audit/stats?limit=30" 
```