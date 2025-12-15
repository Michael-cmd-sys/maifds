# Current Dataset Information

## Dataset: malicious_phish.csv

**Source**: Added by user on Dec 11, 2025

**Statistics:**
- Total URLs in original dataset: 651,191
- Malicious URLs extracted: 223,088
- Unique URLs imported: 211,531

**Breakdown by Type:**
- Phishing URLs: 94,111
- Defacement URLs: 96,457
- Malware URLs: 32,520

**Import Performance:**
- Import time: ~34 seconds
- Method: Optimized bulk import
- Bloom filter size: 1.14 MB
- Load factor: 12.41%

## Files

### Raw Data
- `data/raw/malicious_phish.csv` - Original dataset (45MB)
- `data/raw/blacklist_malicious_urls.csv` - Transformed for import (25MB)

### Processed Data
- `data/blacklist_db.json` - Database with 211,531 URL hashes
- `data/bloom_filters/urls.bloom` - Bloom filter with 223,088 elements
- `data/bloom_filters/numbers.bloom` - Empty (no phone numbers in this dataset)
- `data/bloom_filters/devices.bloom` - Empty (no device IDs in this dataset)

## Usage

### Check a URL
```bash
cd ~/projet/maifds/HUAWEI/Blacklist_Watchlist_Service
~/mindspore311_env/bin/python src/manage_blacklist.py check url "http://suspicious-site.com"
```

### View Statistics
```bash
~/mindspore311_env/bin/python src/manage_blacklist.py stats
```

### Start API
```bash
~/mindspore311_env/bin/python src/api_blacklist.py --port 5001
```

## Adding More Data

To add phone numbers or device IDs, create a CSV with this format:
```csv
type,value,reason,severity,source
phone_number,+237123456789,fraud,high,manual
device_id,IMEI-123456,stolen,critical,telco
url,https://bad.com,phishing,high,scan
```

Then import:
```bash
~/mindspore311_env/bin/python bulk_import.py your_data.csv
```

## Performance

- URL lookup: < 1ms (with Redis cache)
- URL lookup: < 5ms (without Redis)
- Bloom filter false positive rate: 1%
- Memory usage: ~1.5 MB for Bloom filters

---
Last updated: Dec 11, 2025
