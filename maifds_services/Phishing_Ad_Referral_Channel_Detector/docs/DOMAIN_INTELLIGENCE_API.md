# Domain Intelligence API Documentation

## Overview
The Domain Intelligence API provides comprehensive URL and domain analysis powered by WhoisXML API. This service enhances phishing detection with real-time domain intelligence.

## Configuration

Add to `.env.production`:
```bash
WHOISXML_API_KEY=your_api_key_here
```

Get your API key from: https://whoisxmlapi.com/

## API Endpoints

### 1. Comprehensive Domain Analysis
**Endpoint:** `POST /analyze-domain`

Analyzes a URL through multiple intelligence layers and provides a consolidated risk assessment.

**Request:**
```json
{
  "url": "https://suspicious-site.com/page"
}
```

**Response:**
```json
{
  "original_url": "https://suspicious-site.com/page",
  "timestamp": "2026-01-22T10:00:00",
  "expansion": {
    "success": true,
    "final_domain": "suspicious-site.com",
    "was_shortened": false
  },
  "whois": {
    "success": true,
    "domain": "suspicious-site.com",
    "creation_date": "2026-01-15T00:00:00Z",
    "age_days": 7,
    "registrant_name": "Redacted for Privacy",
    "registrant_country": "XX",
    "registrar": "Example Registrar"
  },
  "reputation": {
    "success": true,
    "reputation_score": 25,
    "mode": "suspicious",
    "is_malicious": false,
    "is_suspicious": true
  },
  "geolocation": {
    "success": true,
    "country": "Unknown Country",
    "country_code": "XX",
    "isp": "Shady Hosting Inc"
  },
  "dns": {
    "success": true,
    "has_mx_records": false,
    "mx_count": 0
  },
  "risk_assessment": {
    "level": "HIGH_RISK",
    "signals": [
      "Domain less than 30 days old",
      "Domain flagged as suspicious",
      "No email server configured"
    ],
    "signal_count": 3
  }
}
```

---

### 2. URL Expansion (Unshortening)
**Endpoint:** `POST /expand-url`

Follows short URLs (bit.ly, tinyurl, etc.) to reveal the real destination domain.

**Request:**
```json
{
  "url": "https://bit.ly/abc123"
}
```

**Response:**
```json
{
  "success": true,
  "original_url": "https://bit.ly/abc123",
  "final_url": "https://real-phishing-site.com/steal-momo",
  "final_domain": "real-phishing-site.com",
  "redirect_chain": [
    "https://bit.ly/abc123",
    "https://real-phishing-site.com/steal-momo"
  ],
  "was_shortened": true
}
```

---

### 3. WHOIS Lookup
**Endpoint:** `POST /domain-whois`

Retrieves domain registration data including age and owner information.

**Request:**
```json
{
  "domain": "example.com"
}
```

**Response:**
```json
{
  "success": true,
  "domain": "example.com",
  "creation_date": "1995-08-14T04:00:00Z",
  "age_days": 11118,
  "registrant_name": "Redacted for Privacy",
  "registrant_org": "Example Organization",
  "registrant_country": "US",
  "registrar": "Example Registrar, Inc."
}
```

**Domain Age Risk Levels:**
- **< 30 days:** HIGH RISK (New phishing sites)
- **30-90 days:** MEDIUM RISK (Recently created)
- **> 90 days:** LOW RISK

---

### 4. Domain Reputation Check
**Endpoint:** `POST /domain-reputation`

Checks if the domain is flagged for malicious activity (malware, phishing, spam).

**Request:**
```json
{
  "domain": "known-bad-site.com"
}
```

**Response:**
```json
{
  "success": true,
  "domain": "known-bad-site.com",
  "reputation_score": 15,
  "mode": "malicious",
  "is_malicious": true,
  "is_suspicious": true,
  "test_results": {
    "malware": true,
    "phishing": true,
    "spam": false
  }
}
```

**Reputation Scores:**
- **0-30:** Malicious (Block immediately)
- **31-60:** Suspicious (High scrutiny)
- **61-100:** Safe (Likely legitimate)

---

## Use Cases

### 1. Pre-Transaction Validation
Before allowing a mobile money transaction that originated from an ad click:

```python
import requests

response = requests.post('http://your-api/analyze-domain', json={
    'url': user_clicked_link
})

analysis = response.json()
risk_level = analysis['risk_assessment']['level']

if risk_level == 'HIGH_RISK':
    # Block transaction and alert user
    send_sms_alert(user_phone, "Suspicious link detected!")
elif risk_level == 'MEDIUM_RISK':
    # Require additional authentication
    require_second_factor_auth(user)
```

### 2. Ad Network Screening
Screen ads before displaying them to users:

```python
for ad in pending_ads:
    analysis = analyze_domain(ad.landing_url)
    
    if analysis['whois']['age_days'] < 30:
        reject_ad(ad, reason="Domain too new")
    
    if analysis['reputation']['is_malicious']:
        blacklist_advertiser(ad.advertiser_id)
```

### 3. URL Shortener Protection
Automatically expand shortened URLs in SMS/social media:

```python
if is_shortened_url(url):
    expansion = expand_url(url)
    real_domain = expansion['final_domain']
    
    # Check the REAL domain, not the shortener
    reputation = check_reputation(real_domain)
```

---

## Error Handling

All endpoints return standard error responses:

```json
{
  "error": "Error description",
  "success": false
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad request (missing parameters)
- `500` - Server error
- `503` - Service unavailable (API key issue)

---

## Rate Limiting

WhoisXML API has rate limits based on your plan:
- **Free tier:** 500 requests/month
- **Paid tiers:** 1,000 - 1,000,000/month

The service will return an error if you exceed your quota.

---

## Best Practices

1. **Cache Results:** Domain data doesn't change frequently. Cache WHOIS/reputation data for at least 24 hours.

2. **Batch Analysis:** For bulk screening, collect URLs and analyze in batches to minimize API calls.

3. **Fallback Logic:** If WhoisXML is unavailable, fall back to the free `python-whois` library for basic age checking.

4. **Privacy:** WHOIS data often shows "Redacted for Privacy" due to GDPR. Focus on **domain age** and **reputation** rather than owner names.

---

## Testing

Test the endpoints with known domains:

```bash
# Test expansion
curl -X POST http://localhost:5000/expand-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://bit.ly/3abc"}'

# Test WHOIS on old domain (should show > 10 years)
curl -X POST http://localhost:5000/domain-whois \
  -H "Content-Type: application/json" \
  -d '{"domain": "google.com"}'

# Test comprehensive analysis
curl -X POST http://localhost:5000/analyze-domain \
  -H "Content-Type: application/json" \
  -d '{"url": "https://suspicious-new-site.xyz"}'
```
