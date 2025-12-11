# Real-time Blacklist/Watchlist Service Documentation

## Overview

The **Real-time Blacklist/Watchlist Service** is an ultra-fast fraud detection system that uses **Bloom filters** for O(1) membership testing of known bad phone numbers, device IDs, and URLs.

### Key Features

- **Ultra-Fast Lookups**: O(1) complexity using Bloom filters
- **Low Memory Footprint**: Space-efficient probabilistic data structure
- **Multi-Layer Architecture**: Bloom filter → Redis cache → Authoritative database
- **Real-time Actions**: Auto-block transactions, enforce verification, send notifications
- **Scalable**: Handles millions of entries with minimal memory
- **Persistent Storage**: Database and Bloom filters saved to disk

## Architecture

```
┌─────────────────┐
│   API Request   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Blacklist Service                  │
│                                      │
│  1. Check Redis Cache (O(1))        │
│  2. Check Bloom Filter (O(1))       │
│  3. Verify with Database             │
│  4. Determine Actions                │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Actions                            │
│                                      │
│  • Block Transaction                 │
│  • Enforce High Verification         │
│  • Send Notifications                │
└─────────────────────────────────────┘
```

## Components

### 1. Bloom Filter

A space-efficient probabilistic data structure for set membership testing:
- **Size**: Calculated based on expected elements and desired false positive rate
- **No False Negatives**: If Bloom filter says "not in set", it's definitely not in set
- **Possible False Positives**: If Bloom filter says "in set", verify with database
- **Memory Efficient**: ~1.44 MB per 100,000 entries (at 1% FPR)

### 2. Redis Cache

Ultra-fast in-memory cache for frequently checked entries:
- **O(1) Lookup**: Fastest possible check
- **TTL Support**: Automatic expiry of entries
- **Optional**: Service works without Redis (falls back to Bloom filter + DB)

### 3. Authoritative Database

JSON-based persistent storage:
- **Complete Entry Details**: Reason, severity, source, timestamps
- **Metadata**: Additional information about blacklisted entries
- **Backup**: Can rebuild Bloom filters from database

## Installation

### 1. Install Dependencies

```bash
# Install Redis (optional but recommended)
sudo apt-get install redis-server
sudo systemctl start redis-server

# Install Python dependencies
pip install redis bitarray pandas flask flask-cors
```

### 2. Configure Service

Edit `config_blacklist.json`:

```json
{
  "bloom_filter": {
    "expected_elements": 1000000,
    "false_positive_rate": 0.01
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "cache_expiry": 86400
  },
  "storage": {
    "database_path": "data/blacklist_db.json",
    "bloom_filters_dir": "data/bloom_filters/"
  },
  "actions": {
    "auto_block": true,
    "enforce_high_verification": true,
    "send_notifications": true
  }
}
```

## Usage

### Command Line Interface

#### 1. Add Single Entry

```bash
python manage_blacklist.py add phone_number "+237650123456" \
  --reason fraud \
  --severity high \
  --source fraud_ops
```

#### 2. Import from CSV

```bash
python manage_blacklist.py import data/raw/sample_blacklist.csv
```

CSV Format:
```csv
type,value,reason,severity,source
phone_number,+237650000001,fraud,high,fraud_ops
device_id,IMEI-123456789012345,stolen_device,critical,telco
url,https://phishing-site.xyz/login,phishing,high,security_scan
```

#### 3. Check Entry

```bash
python manage_blacklist.py check phone_number "+237650123456"
```

Output:
```
✗ BLACKLISTED
Matches: ['phone_number']
Severity: high
Actions: ['BLOCK_TRANSACTION', 'ENFORCE_HIGH_VERIFICATION', 'SEND_NOTIFICATION']
```

#### 4. View Statistics

```bash
python manage_blacklist.py stats
```

Output:
```
======================================================================
BLACKLIST/WATCHLIST SERVICE STATISTICS
======================================================================

Detection Statistics:
  Total Checks: 1523
  Total Hits: 47
  Phone Number Hits: 23
  Device ID Hits: 15
  URL Hits: 9
  Blocked Transactions: 47

Database:
  Total Entries: 10523

Bloom Filter - Phone Numbers:
  Size: 1.44 MB
  Elements: 5234
  Load Factor: 0.0876
  Expected FPR: 0.010000
  Actual FPR: 0.009876

Redis Cache:
  Enabled: True
======================================================================
```

#### 5. Export Blacklist

```bash
python manage_blacklist.py export output/blacklist_export.csv
```

#### 6. Rebuild Bloom Filters

After many deletions, rebuild to optimize:

```bash
python manage_blacklist.py rebuild
```

### REST API

#### Start API Server

```bash
python api_blacklist.py --host 0.0.0.0 --port 5001
```

#### API Endpoints

##### 1. Check Blacklist

**POST** `/check`

Request:
```json
{
  "phone_number": "+237650123456",
  "device_id": "IMEI-123456789012345",
  "url": "https://example.com/page",
  "transaction_id": "TXN_20251211_001"
}
```

Response:
```json
{
  "is_blacklisted": true,
  "matches": ["phone_number", "device_id"],
  "severity": "high",
  "actions": [
    "BLOCK_TRANSACTION",
    "ENFORCE_HIGH_VERIFICATION",
    "SEND_NOTIFICATION_TO_FRAUD_OPS",
    "SEND_NOTIFICATION_TO_USER"
  ],
  "details": {},
  "timestamp": "2025-12-11T12:30:00.123456",
  "transaction_id": "TXN_20251211_001"
}
```

##### 2. Batch Check

**POST** `/batch-check`

Request:
```json
{
  "checks": [
    {"phone_number": "+237650000001"},
    {"device_id": "IMEI-123456"},
    {"url": "https://example.com"}
  ]
}
```

Response:
```json
{
  "results": [
    {"index": 0, "is_blacklisted": true, "matches": ["phone_number"]},
    {"index": 1, "is_blacklisted": false, "matches": []},
    {"index": 2, "is_blacklisted": true, "matches": ["url"]}
  ],
  "summary": {
    "total": 3,
    "blacklisted": 2,
    "clean": 1
  }
}
```

##### 3. Add to Blacklist

**POST** `/add`

Request:
```json
{
  "type": "phone_number",
  "value": "+237650123456",
  "reason": "fraud",
  "severity": "high",
  "source": "fraud_ops",
  "additional_info": {
    "reported_by": "user_complaint",
    "num_reports": 5
  }
}
```

Response:
```json
{
  "success": true,
  "hash": "a1b2c3d4e5f6...",
  "type": "phone_number",
  "message": "Entry added to blacklist"
}
```

##### 4. Batch Add

**POST** `/batch-add`

Request:
```json
{
  "entries": [
    {
      "type": "phone_number",
      "value": "+237650000001",
      "reason": "fraud",
      "severity": "high",
      "source": "fraud_ops"
    },
    {
      "type": "url",
      "value": "https://phishing.xyz",
      "reason": "phishing",
      "severity": "critical",
      "source": "security_scan"
    }
  ]
}
```

##### 5. Remove from Blacklist

**POST** `/remove`

Request:
```json
{
  "type": "phone_number",
  "value": "+237650123456"
}
```

##### 6. Get Statistics

**GET** `/statistics`

##### 7. Rebuild Bloom Filters

**POST** `/rebuild`

##### 8. Health Check

**GET** `/health`

### Python Integration

```python
from blacklist_watchlist_service import BlacklistWatchlistService

# Initialize service
service = BlacklistWatchlistService(config_path='config_blacklist.json')

# Add entry
service.add_to_blacklist(
    'phone_number',
    '+237650123456',
    {
        'reason': 'fraud',
        'severity': 'high',
        'source': 'fraud_ops'
    }
)

# Check signals
signals = {
    'phone_number': '+237650123456',
    'device_id': 'IMEI-123456789012345',
    'url': 'https://suspicious-site.com'
}

result = service.check_blacklist(signals)

if result['is_blacklisted']:
    print(f"ALERT: Blacklisted! Matches: {result['matches']}")
    print(f"Actions to take: {result['actions']}")

    # Take appropriate actions
    if 'BLOCK_TRANSACTION' in result['actions']:
        # Block the transaction
        pass

    if 'SEND_NOTIFICATION' in result['actions']:
        # Send alert to fraud ops
        pass
```

## Signals

### Supported Entry Types

1. **phone_number**: Mobile phone numbers (normalized with country code)
2. **device_id**: Device identifiers (IMEI, MAC address, etc.)
3. **url**: URLs and domains

### Hash Generation

All entries are hashed using SHA-256 before storage:
- **Privacy**: Original values are not stored (only hashes)
- **Security**: One-way hashing prevents reverse lookup
- **Consistency**: Same value always produces same hash

## Actions

When a blacklist hit is detected, the service can trigger:

### 1. Block Transaction
- Immediately reject the transaction
- Prevent fund transfer
- Log the attempt

### 2. Enforce High Verification
- Require additional authentication steps
- Request OTP verification
- Increase fraud score

### 3. Send Notifications
- **To Fraud Ops**: Alert fraud detection team via webhook
- **To User**: SMS/email warning about suspicious activity

### 4. Custom Actions
Add custom actions in the code based on your requirements.

## Performance

### Bloom Filter Performance

| Entries | Size (MB) | Lookup Time | FPR (1%) |
|---------|-----------|-------------|----------|
| 100K    | 1.44      | < 1 µs      | 0.01     |
| 1M      | 14.4      | < 1 µs      | 0.01     |
| 10M     | 144       | < 1 µs      | 0.01     |

### System Performance

- **Check Latency**: < 1ms (with Redis cache)
- **Check Latency**: < 5ms (without Redis)
- **Throughput**: > 10,000 checks/second
- **Memory**: ~15MB per 1M entries

## Deployment

### Production Deployment

```bash
# Install production server
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5001 api_blacklist:app

# With systemd service
sudo systemctl enable blacklist-service
sudo systemctl start blacklist-service
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_mindspore.txt .
RUN pip install --no-cache-dir -r requirements_mindspore.txt

COPY src/ ./src/
COPY config_blacklist.json .

EXPOSE 5001

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5001", "src.api_blacklist:app"]
```

### Redis Setup

```bash
# Install Redis
sudo apt-get install redis-server

# Configure Redis for production
sudo nano /etc/redis/redis.conf

# Recommended settings:
maxmemory 256mb
maxmemory-policy allkeys-lru

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## Monitoring

### Key Metrics to Monitor

1. **Detection Statistics**
   - Total checks
   - Hit rate
   - False positive rate

2. **Performance Metrics**
   - Check latency (p50, p95, p99)
   - Throughput (checks/second)
   - Redis hit rate

3. **Bloom Filter Health**
   - Load factor
   - Actual FPR vs expected
   - Memory usage

4. **System Health**
   - Redis availability
   - Disk space for database
   - API response time

### Logging

Logs are written to:
- `blacklist_watchlist_service.log`
- Console output

Configure logging level in the code or via environment variable.

## Maintenance

### Regular Tasks

1. **Backup Database**
   ```bash
   cp data/blacklist_db.json backup/blacklist_db_$(date +%Y%m%d).json
   ```

2. **Monitor Bloom Filter Load**
   ```bash
   python manage_blacklist.py stats
   ```

3. **Rebuild Bloom Filters** (after many deletions)
   ```bash
   python manage_blacklist.py rebuild
   ```

4. **Clear Redis Cache** (if needed)
   ```bash
   redis-cli FLUSHDB
   ```

## Troubleshooting

### Issue: High False Positive Rate

**Solution**: Reduce FPR in config or rebuild with larger Bloom filter
```json
{
  "bloom_filter": {
    "expected_elements": 1000000,
    "false_positive_rate": 0.001
  }
}
```

### Issue: Redis Connection Failed

**Solution**: Service works without Redis. Check if Redis is running:
```bash
sudo systemctl status redis-server
redis-cli ping
```

### Issue: Slow Performance

**Solutions**:
1. Enable Redis cache
2. Increase Redis memory
3. Optimize Bloom filter size
4. Use SSD for database storage

## Best Practices

1. **Regular Backups**: Backup database daily
2. **Monitor FPR**: Keep actual FPR close to expected
3. **Rebuild Periodically**: Rebuild Bloom filters monthly
4. **Use Redis**: Enable Redis for best performance
5. **Set TTL**: Use cache expiry to keep data fresh
6. **Log Everything**: Monitor all blacklist hits
7. **Test Regularly**: Test API endpoints and CLI tools

## Security Considerations

1. **API Authentication**: Add API key authentication in production
2. **Rate Limiting**: Implement rate limiting on API endpoints
3. **HTTPS**: Use HTTPS in production
4. **Hash Salting**: Consider adding salt to hashes for extra security
5. **Access Control**: Restrict who can add/remove entries

## Integration with MoMo Platform

```python
# In your transaction processing code:

from blacklist_watchlist_service import BlacklistWatchlistService

service = BlacklistWatchlistService()

def process_transaction(transaction):
    # Extract signals
    signals = {
        'phone_number': transaction.sender_phone,
        'device_id': transaction.device_id
    }

    # Check blacklist
    result = service.check_blacklist(signals)

    if result['is_blacklisted']:
        # Block transaction
        if 'BLOCK_TRANSACTION' in result['actions']:
            return {
                'status': 'blocked',
                'reason': 'blacklisted',
                'matches': result['matches']
            }

        # Or enforce high verification
        if 'ENFORCE_HIGH_VERIFICATION' in result['actions']:
            return {
                'status': 'pending_verification',
                'verification_level': 'high'
            }

    # Process normally
    return process_transaction_normally(transaction)
```

## Support

For issues or questions:
- Check the logs: `blacklist_watchlist_service.log`
- Review configuration: `config_blacklist.json`
- Test CLI tools: `python manage_blacklist.py stats`
- Check API health: `curl http://localhost:5001/health`

## References

- Bloom Filter Theory: https://en.wikipedia.org/wiki/Bloom_filter
- Redis Documentation: https://redis.io/documentation
- Flask Documentation: https://flask.palletsprojects.com/
