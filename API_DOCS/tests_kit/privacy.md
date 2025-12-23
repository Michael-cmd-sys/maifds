## 📌 Canonical 10 cURL tests

- `These are the official, correct tests for our documentation and demo.`

## 1. Health check

```bash
curl -s http://127.0.0.1:8000/v1/governance/privacy/health
```

## 2. Classify email (PII)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"Email me at test@example.com"}'

```

## 3. Classify phone number

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"Call +233554123456 now"}'

```

## 4. Classify email + phone

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"Call +233554123456 or email test@example.com"}'

```

## 5. Classify IP address

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"User logged in from 197.0.0.1"}'

```

## 6. Classify credit card

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"Payment card 4111 1111 1111 1111 was used"}'

```

## 7. Classify SSN

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \
  -H "Content-Type: application/json" \
  -d '{"text":"SSN 123-45-6789 was submitted"}'

```

## 8. Anonymize (mask)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/anonymize" \
  -H "Content-Type: application/json" \
  -d '{"text":"Call +233554123456 or email test@example.com","strategy":"mask"}'

```

## 9. Anonymize IP + card

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/anonymize" \
  -H "Content-Type: application/json" \
  -d '{"text":"IP 197.0.0.1 used card 4111 1111 1111 1111","strategy":"mask"}'

```

## 10. Anonymize (hash)

```bash
curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/anonymize" \
  -H "Content-Type: application/json" \
  -d '{"text":"test@example.com +233554123456 197.0.0.1","strategy":"hash"}'

```