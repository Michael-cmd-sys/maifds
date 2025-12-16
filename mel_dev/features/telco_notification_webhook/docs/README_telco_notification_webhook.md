# Telco Notification Webhook

## Overview

The **Telco Notification Webhook** feature is a non-ML integration component of the
MAIFDS platform. It is responsible for **sending structured fraud incidents**
to external telco, SIEM, or fraud-operations systems via authenticated HTTP webhooks.

This module does **not perform inference or training**.
It acts as a downstream incident delivery and audit layer.

---

## Responsibilities

- Accept fraud incidents from internal services
- Send incidents to a configured telco webhook endpoint
- Retry on network or 5xx failures
- Log all incidents and webhook responses for auditability

---

## Architecture


```
FastAPI (/v1/telco-notify)
|
v
TelcoNotificationClient
|
v
External Telco / SIEM Webhook
```

---

## API Endpoint

### `POST /v1/telco-notify`

#### Request Body (minimum)

```json
{
  "incident_id": "inc_001",
  "suspected_number": "+233201234567",
  "recommended_action": "BLOCK_NUMBER"
}
```

Optional Fields

affected_accounts: list of impacted wallet or account IDs

observed_evidence: free-form evidence dictionary

timestamps: event timing metadata

Payload Handling

Internally, the client supports:

TelcoIncidentPayload schema objects

Plain dictionaries from the API gateway

This allows flexible integration without forcing external callers
to construct internal schemas.

Configuration

**Environment variables (optional):**


- Variable	Description
- TELCO_WEBHOOK_URL	Target webhook endpoint
- TELCO_WEBHOOK_API_KEY	Bearer token for auth
- TELCO_WEBHOOK_TIMEOUT	Request timeout (seconds)
- TELCO_WEBHOOK_MAX_RETRIES	Retry count
- FRAUD_PLATFORM_ENV	dev / staging / prod
- Logging


All activity is logged locally for audit and troubleshooting:

```
data/logs/
├── incidents.jsonl
└── webhook_errors.jsonl
```


**Each record includes:**

- Timestamp

- Payload sent

- Telco response or error

- Error Handling

- Network errors → retried with backoff

- Non-2xx responses → logged and retried

- Final failure → recorded and returned as response

- Security Notes

- No PII beyond phone numbers should be sent

- Evidence should be hashed or tokenized

- API keys must be stored securely in production