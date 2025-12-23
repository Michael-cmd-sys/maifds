## Quick Test Run
**1) Confirm it was added (check URL)**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/check \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://evil-site.xyz"
  }'

```

**2) Get stats again (see counters change after checks)**

```bash
curl -s http://127.0.0.1:8000/v1/blacklist/stats
```

**3) Remove it (cleanup)**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/remove \
  -H "Content-Type: application/json" \
  -d '{
    "list_type": "url",
    "value": "https://evil-site.xyz"
  }'

```

## The 10 curl tests (updated: NO jq)

**1) Stats**

- `curl -s http://127.0.0.1:8000/v1/blacklist/stats`


**2) Add URL**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/add -H "Content-Type: application/json" -d '{
  "list_type": "url",
  "value": "https://evil-site.xyz",
  "metadata": {"reason":"phishing","severity":"high","source":"manual"}
}'
```


**3) Check URL**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/check -H "Content-Type: application/json" -d '{
  "url": "https://evil-site.xyz"
}'
```


**4) Remove URL**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/remove -H "Content-Type: application/json" -d '{
  "list_type": "url",
  "value": "https://evil-site.xyz"
}'
```


**5) Check URL again**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/check -H "Content-Type: application/json" -d '{
  "url": "https://evil-site.xyz"
}'
```


**6) Add phone**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/add -H "Content-Type: application/json" -d '{
  "list_type": "phone_number",
  "value": "+233201234567",
  "metadata": {"reason":"fraud_reports","severity":"high","source":"fraud_ops"}
}'
```


**7) Check phone**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/check -H "Content-Type: application/json" -d '{
  "phone_number": "+233201234567"
}'
```


**8) Add device**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/add -H "Content-Type: application/json" -d '{
  "list_type": "device_id",
  "value": "IMEI-123456789012345",
  "metadata": {"reason":"stolen_device","severity":"critical","source":"telco"}
}'
```


**9) Check multi-signal**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/check -H "Content-Type: application/json" -d '{
  "phone_number": "+233201234567",
  "device_id": "IMEI-123456789012345",
  "url": "https://evil-site.xyz"
}'
```


**10) Rebuild bloom filters**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/blacklist/rebuild \
  -H "Content-Type: application/json" \
  -d '{}'
```  