# User SMS Alert Service (MAIFDS)

This feature sends **real-time SMS warnings to a user** when MAIFDS detects suspicious mobile money fraud activity.  
It uses the **Moolre SMS API** and supports Ghana phone numbers in both formats:

- `0xxxxxxxxx`
- `+233xxxxxxxxx`

The service also logs all sent messages (and errors) to JSONL files so you can build a dashboard showing:
- who received what message
- the exact SMS text sent
- the provider response (success/failure)
- refs (request ids) for traceability

---

## What this feature does

### 1) Phone normalization
Incoming numbers are normalized into **digits-only E.164 Ghana format**:

- `0xxxxxxxxx` → `233xxxxxxxxx`
- `+233xxxxxxxxx` → `233xxxxxxxxx`

This ensures consistent storage + provider compatibility.

### 2) Message building
You can either:
- provide a custom `message` in the API request, or
- provide a `threat_type` and `context`, and let MAIFDS generate the SMS using templates (e.g. HIGH/MEDIUM scam warning).

### 3) Send SMS via Moolre
The service calls:

`POST https://api.moolre.com/open/sms/send`

Payload format:

```json
{
  "type": 1,
  "senderid": "YOUR_SENDER_ID",
  "messages": [
    {
      "recipient": "233559426442",
      "message": "MAIFDS ALERT: ...",
      "ref": "orch-0001"
    }
  ]
}
```

**4) Logging for dashboard**

## Every send attempt is logged to JSONL for dashboard visibility.

## Folder structure

## Typical layout:

```
mel_dev/features/user_sms_alert/
  data/
    logs/
      sms_sent.jsonl
      sms_errors.jsonl
  docs/
    README_user_sms_alert.md
  src/
    client.py
    config.py
    schemas.py
    storage.py
    phone.py
    message_builder.py
```

**Environment variables (.env)**

## Put these in your repo .env (and do NOT commit it):

```
MOOLRE_X_API_VASKEY=xxxxxxxxxxxxxxxx

MOOLRE_SENDER_ID=YOUR_SENDER_ID

MOOLRE_SMS_SEND_URL=https://api.moolre.com/open/sms/send (optional, defaults to Moolre)

MOOLRE_VERIFY_SSL=true (optional)

MOOLRE_TIMEOUT=5.0 (optional)

MOOLRE_MAX_RETRIES=3 (optional)
```
