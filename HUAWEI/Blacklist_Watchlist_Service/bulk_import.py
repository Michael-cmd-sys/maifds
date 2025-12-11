"""
Optimized bulk import for large datasets
Imports directly without saving after each entry
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, 'src')

from blacklist_watchlist_service import BlacklistWatchlistService, HashGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bulk_import_optimized(csv_path: str):
    """Fast bulk import for large datasets"""

    logger.info(f"Starting optimized bulk import from: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Initialize service
    service = BlacklistWatchlistService()
    hash_gen = HashGenerator()

    # Prepare all entries first
    logger.info("Processing entries...")
    entries_by_type = {
        'phone_numbers': [],
        'device_ids': [],
        'urls': []
    }

    for idx, row in df.iterrows():
        entry_type = row['type']
        value = row['value']

        # Generate hash
        if entry_type == 'phone_number':
            hash_value = hash_gen.hash_phone_number(value)
            type_key = 'phone_numbers'
            bloom_filter = service.bloom_filter_numbers
        elif entry_type == 'device_id':
            hash_value = hash_gen.hash_device_id(value)
            type_key = 'device_ids'
            bloom_filter = service.bloom_filter_devices
        elif entry_type == 'url':
            hash_value = hash_gen.hash_url(value)
            type_key = 'urls'
            bloom_filter = service.bloom_filter_urls
        else:
            continue

        # Add to Bloom filter
        bloom_filter.add(hash_value)

        # Prepare database entry
        metadata = {
            'added_date': datetime.now().isoformat(),
            'reason': row.get('reason', 'unknown'),
            'severity': row.get('severity', 'high'),
            'source': row.get('source', 'bulk_import'),
            'additional_info': {}
        }

        entries_by_type[type_key].append((hash_value, metadata))

        # Show progress every 10k entries
        if (idx + 1) % 10000 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} entries...")

    logger.info(f"Finished processing all {len(df)} entries")

    # Add all entries to database at once
    logger.info("Adding entries to database...")
    for entry_type, entries in entries_by_type.items():
        logger.info(f"  Adding {len(entries)} {entry_type}...")
        for hash_value, metadata in entries:
            service.database.data[entry_type][hash_value] = metadata

    # Save database once
    logger.info("Saving database...")
    service.database._save_database()

    # Save Bloom filters
    logger.info("Saving Bloom filters...")
    service.save_bloom_filters()

    # Print statistics
    logger.info("\n" + "="*70)
    logger.info("IMPORT COMPLETE")
    logger.info("="*70)
    logger.info(f"Total entries imported: {len(df)}")
    logger.info(f"  Phone numbers: {len(entries_by_type['phone_numbers'])}")
    logger.info(f"  Device IDs: {len(entries_by_type['device_ids'])}")
    logger.info(f"  URLs: {len(entries_by_type['urls'])}")
    logger.info("="*70)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python bulk_import.py <csv_file>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    try:
        success = bulk_import_optimized(csv_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
