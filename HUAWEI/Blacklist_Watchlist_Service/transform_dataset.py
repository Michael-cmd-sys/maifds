"""
Transform malicious_phish.csv to blacklist import format
"""

import pandas as pd
import sys

print("Loading malicious_phish.csv...")
df = pd.read_csv('data/raw/malicious_phish.csv')

print(f"Total rows: {len(df)}")
print(f"\nDistribution:")
print(df['type'].value_counts())

# Filter only malicious URLs (exclude benign)
malicious = df[df['type'].isin(['phishing', 'defacement', 'malware'])].copy()

print(f"\nMalicious URLs: {len(malicious)}")

# Transform to blacklist format
# Columns needed: type, value, reason, severity, source
blacklist_data = []

for idx, row in malicious.iterrows():
    url = row['url']
    url_type = row['type']

    # Map type to severity
    severity_map = {
        'phishing': 'critical',
        'malware': 'critical',
        'defacement': 'high'
    }

    blacklist_data.append({
        'type': 'url',  # All entries are URLs
        'value': url if url.startswith('http') else f'http://{url}',
        'reason': url_type,
        'severity': severity_map.get(url_type, 'high'),
        'source': 'malicious_phish_dataset'
    })

# Create DataFrame
blacklist_df = pd.DataFrame(blacklist_data)

# Save to CSV
output_file = 'data/raw/blacklist_malicious_urls.csv'
blacklist_df.to_csv(output_file, index=False)

print(f"\n✅ Transformed dataset saved to: {output_file}")
print(f"   Total malicious URLs: {len(blacklist_df)}")
print(f"\n   Breakdown:")
print(f"   - Phishing: {len(blacklist_df[blacklist_df['reason'] == 'phishing'])}")
print(f"   - Defacement: {len(blacklist_df[blacklist_df['reason'] == 'defacement'])}")
print(f"   - Malware: {len(blacklist_df[blacklist_df['reason'] == 'malware'])}")
