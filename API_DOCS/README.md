# Fraud Defense API Docs

This document describes the Fraud Defense System API endpoints, request/response formats, risk interpretation, and test kits.

Base URL (local):
- `http://127.0.0.1:8000`

API Prefix:
- `/v1`

---

## Conventions

### Risk Levels
- **LOW**: allow transaction, log only
- **MEDIUM**: warn user, step-up confirmation (prompt)
- **HIGH**: block/hold transaction, alert user, notify telco/security

### Actions (standard)
These are canonical action codes returned by feature engines.

- `ALLOW`
- `LOG_ONLY`
- `USER_PROMPT`
- `STEP_UP_AUTH`
- `SMS_USER_ALERT`
- `TEMP_HOLD`
- `BLOCK_TX`
- `NOTIFY_TELCO`
- `NOTIFY_SECURITY_TEAM`

> Not every feature must return all actions; return only what applies.

### Response Wrapper
All feature endpoints follow this structure:
```
```json
{
  "feature": "<feature_name>",
  "result": { ...feature_specific_result... }
}
```

Error Format (FastAPI default)

Validation errors typically return HTTP 422 with details.

Health & Info (recommended)
GET /v1/health

Purpose: confirm API is running.

Response (example):

```
{ "status": "ok" }
```

Feature 1: Call-Triggered Defense

Detects high-risk fraud patterns where a user receives a call and quickly performs a transaction (common scam flow). Uses a hybrid approach:

ML model score (MindSpore inference)

Rule-based override for high-precision scam patterns

Endpoint:

POST /v1/call-triggered-defense

Request Schema

Content-Type:

application/json

Body fields:

| Field                     | Type    | Required | Notes                                                     |
|---------------------------|---------|----------|-----------------------------------------------------------|
| tx_amount                 | number  | ✅       | Transaction amount (same unit as your system; e.g., GHS) |
| recipient_first_time      | integer | ✅       | `1` if first time sending to recipient, else `0`         |
| call_to_tx_delta_seconds  | integer | ✅       | Seconds between call and transaction                      |
| contact_list_flag         | integer | ✅       | `1` if recipient is in contacts, else `0`                |
| nlp_suspicion_score       | number  | ✅       | `0.0`–`1.0` suspicion score from call/chat analysis      |


Example:

{
  "tx_amount": 5000,
  "recipient_first_time": 1,
  "call_to_tx_delta_seconds": 25,
  "contact_list_flag": 0,
  "nlp_suspicion_score": 0.7
}

Response Schema

**Success: HTTP 200**

```
Body fields:

Field	Type	Description
feature	string	Always "call_triggered_defense"
result.fraud_probability	number	ML model probability (0.0–1.0)
result.rule_flag	boolean	true if a high-precision rule triggered
result.risk_level	string	LOW / MEDIUM / HIGH
result.reason	string	Human-readable explanation
result.actions	string[]	Action codes to execute
```

**Example response:**
```
{
  "feature": "call_triggered_defense",
  "result": {
    "fraud_probability": 0.0,
    "rule_flag": true,
    "risk_level": "HIGH",
    "reason": "Rule-based high-risk pattern detected",
    "actions": ["SMS_USER_ALERT", "TEMP_HOLD", "NOTIFY_TELCO"]
  }
}
```

How Final Risk Is Determined (High-Level)

The engine combines:

ML model score (fraud_probability)

Rules (override/escalation)

Typical behavior:

If a high-precision rule triggers → risk_level becomes HIGH even if ML score is low.

If no rule triggers → risk may be based on ML thresholding.

This design protects against cold-start ML models and ensures obvious scam patterns are caught.


Notes for Developers
Common Gotchas

Ensure feature scaling/normalization in inference matches training.

If ML always returns ~0.0, verify:

- `correct model checkpoint path`

- `model is in eval mode`

- `inputs are normalized`

- `feature order matches **FEATURE_COLUMNS**`

### Logging (recommended)

**Log at minimum:**

- `request ID`

- `selected rules triggered`

- `model score`

- `final risk level`

- `actions`

**Feature Registry**

## Feature 1: Call-Triggered Defense ✅

Real-time fraud prevention triggered by telecom call events, combining ML scoring and deterministic rules to block or hold high-risk transactions immediately after suspicious calls.

## Feature 2: Phishing Ad & Referral Channel Detector ✅

Detection of malicious ads, referral links, and phishing entry points using URL analysis, reputation signals, and blacklist/watchlist correlation.

## Feature 3: Click-to-Transaction Link Correlation & Blocker ✅

Correlates user clicks, calls, and navigation events with downstream transactions to detect scam funnels and block transactions linked to suspicious interaction paths.

## Feature 4: Customer Reporting & Crowd-Sourced Reputation System ✅

User-driven reporting of scams, bad actors, and suspicious entities, feeding a shared reputation system that influences real-time risk scoring.

## Feature 5: Real-Time Blacklist / Watchlist Service (Bloom Filters) ✅

Ultra-low-latency screening of phone numbers, URLs, IPs, wallets, and accounts using Bloom filters for hot-path transaction gating.

## Feature 6: Agent / Merchant Risk Profiling & Mule Network Detection ✅

Behavioral profiling and network-level analysis to identify high-risk agents, merchants, and mule networks through transaction patterns and shared attributes.

## Feature 7: Human-in-the-Loop Alerting & Verification Portal ✅

Operational dashboard for analysts and compliance teams to review alerts, verify cases, override decisions, and provide feedback to models.

## Feature 8: Proactive Pre-Transaction Warning & User Prompting ✅

Risk-aware user warnings, confirmations, or step-up checks issued before transaction execution to prevent social-engineering fraud.

## Feature 9: Automated Telco Notification & Triage Webhook ✅

Automated escalation of confirmed or high-risk cases to telecom partners via webhooks for call blocking, SIM monitoring, or further investigation.

## Feature 10: Automated User Notification via SMS ✅

Real-time SMS alerts to users for suspicious activity, blocked transactions, or confirmation requests, closing the loop with the customer.

## Feature 11: Explainability, Audit Trail & Legal / Privacy Controls ✅

End-to-end explainability, immutable audit trails, event correlation, privacy-aware data classification, anonymization, consent management, and GDPR-aligned controls.