"""
Management script for Real-time Blacklist/Watchlist Service
Handles bulk import, export, and maintenance operations
"""

import sys
import os
import json
import csv
import argparse
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd

from blacklist_watchlist_service import BlacklistWatchlistService, HashGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_from_csv(service: BlacklistWatchlistService, csv_path: str):
    """
    Import blacklist entries from CSV file

    Expected CSV format:
    type,value,reason,severity,source
    phone_number,+1234567890,fraud,high,manual
    device_id,IMEI-123456,stolen,critical,telco
    url,https://bad.com,phishing,high,scan
    """
    logger.info(f"Importing from CSV: {csv_path}")

    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"CSV loaded: {len(df)} rows")

        required_columns = ['type', 'value', 'reason', 'severity', 'source']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV must have columns: {required_columns}")
            return False

        # Prepare batch entries
        entries = []
        for _, row in df.iterrows():
            metadata = {
                'reason': row['reason'],
                'severity': row['severity'],
                'source': row['source'],
                'additional_info': {}
            }

            # Add any extra columns as additional info
            for col in df.columns:
                if col not in required_columns:
                    metadata['additional_info'][col] = row[col]

            entries.append((row['type'], row['value'], metadata))

        # Batch add
        service.batch_add_to_blacklist(entries)

        logger.info(f"Successfully imported {len(entries)} entries")
        return True

    except Exception as e:
        logger.error(f"Error importing CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_to_csv(service: BlacklistWatchlistService, csv_path: str):
    """Export blacklist entries to CSV file"""
    logger.info(f"Exporting to CSV: {csv_path}")

    try:
        rows = []

        # Export phone numbers
        for hash_value, entry in service.database.data.get('phone_numbers', {}).items():
            rows.append({
                'hash': hash_value,
                'type': 'phone_number',
                'reason': entry.get('reason', ''),
                'severity': entry.get('severity', ''),
                'source': entry.get('source', ''),
                'added_date': entry.get('added_date', ''),
                'additional_info': json.dumps(entry.get('additional_info', {}))
            })

        # Export device IDs
        for hash_value, entry in service.database.data.get('device_ids', {}).items():
            rows.append({
                'hash': hash_value,
                'type': 'device_id',
                'reason': entry.get('reason', ''),
                'severity': entry.get('severity', ''),
                'source': entry.get('source', ''),
                'added_date': entry.get('added_date', ''),
                'additional_info': json.dumps(entry.get('additional_info', {}))
            })

        # Export URLs
        for hash_value, entry in service.database.data.get('urls', {}).items():
            rows.append({
                'hash': hash_value,
                'type': 'url',
                'reason': entry.get('reason', ''),
                'severity': entry.get('severity', ''),
                'source': entry.get('source', ''),
                'added_date': entry.get('added_date', ''),
                'additional_info': json.dumps(entry.get('additional_info', {}))
            })

        # Write to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        logger.info(f"Successfully exported {len(rows)} entries to {csv_path}")
        return True

    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return False


def add_single_entry(service: BlacklistWatchlistService, entry_type: str, value: str,
                     reason: str, severity: str, source: str):
    """Add a single entry to blacklist"""
    metadata = {
        'reason': reason,
        'severity': severity,
        'source': source,
        'additional_info': {}
    }

    result = service.add_to_blacklist(entry_type, value, metadata)

    if result['success']:
        logger.info(f"Added to blacklist: {entry_type} - {value}")
        service.save_bloom_filters()
        return True
    else:
        logger.error(f"Failed to add: {result.get('error')}")
        return False


def remove_single_entry(service: BlacklistWatchlistService, entry_type: str, value: str):
    """Remove a single entry from blacklist"""
    result = service.remove_from_blacklist(entry_type, value)

    if result['success']:
        logger.info(f"Removed from blacklist: {entry_type} - {value}")
        return True
    else:
        logger.error(f"Failed to remove: {result.get('error')}")
        return False


def check_entry(service: BlacklistWatchlistService, entry_type: str, value: str):
    """Check if entry is blacklisted"""
    signals = {entry_type: value}
    result = service.check_blacklist(signals)

    if result['is_blacklisted']:
        print(f"\n✗ BLACKLISTED")
        print(f"Matches: {result['matches']}")
        print(f"Severity: {result['severity']}")
        print(f"Actions: {result['actions']}")
    else:
        print(f"\n✓ NOT BLACKLISTED")

    return result['is_blacklisted']


def show_statistics(service: BlacklistWatchlistService):
    """Show service statistics"""
    stats = service.get_statistics()

    print("\n" + "="*70)
    print("BLACKLIST/WATCHLIST SERVICE STATISTICS")
    print("="*70)

    print("\nDetection Statistics:")
    print(f"  Total Checks: {stats['detection_stats']['total_checks']}")
    print(f"  Total Hits: {stats['detection_stats']['total_hits']}")
    print(f"  Phone Number Hits: {stats['detection_stats']['phone_hits']}")
    print(f"  Device ID Hits: {stats['detection_stats']['device_hits']}")
    print(f"  URL Hits: {stats['detection_stats']['url_hits']}")
    print(f"  Blocked Transactions: {stats['detection_stats']['blocked_transactions']}")

    print("\nDatabase:")
    print(f"  Total Entries: {stats['database_entries']}")

    print("\nBloom Filter - Phone Numbers:")
    bf_nums = stats['bloom_filter_numbers']
    print(f"  Size: {bf_nums['size_mb']:.2f} MB")
    print(f"  Elements: {bf_nums['num_elements']}")
    print(f"  Load Factor: {bf_nums['load_factor']:.4f}")
    print(f"  Expected FPR: {bf_nums['expected_fpr']:.6f}")
    print(f"  Actual FPR: {bf_nums['actual_fpr']:.6f}")

    print("\nBloom Filter - Device IDs:")
    bf_devs = stats['bloom_filter_devices']
    print(f"  Size: {bf_devs['size_mb']:.2f} MB")
    print(f"  Elements: {bf_devs['num_elements']}")
    print(f"  Load Factor: {bf_devs['load_factor']:.4f}")

    print("\nBloom Filter - URLs:")
    bf_urls = stats['bloom_filter_urls']
    print(f"  Size: {bf_urls['size_mb']:.2f} MB")
    print(f"  Elements: {bf_urls['num_elements']}")
    print(f"  Load Factor: {bf_urls['load_factor']:.4f}")

    print("\nRedis Cache:")
    print(f"  Enabled: {stats['redis_cache_enabled']}")

    print("="*70 + "\n")


def rebuild_filters(service: BlacklistWatchlistService):
    """Rebuild Bloom filters from database"""
    logger.info("Rebuilding Bloom filters...")
    service.rebuild_bloom_filters()
    logger.info("Bloom filters rebuilt successfully")


def main():
    """Main management function"""
    parser = argparse.ArgumentParser(
        description='Manage Real-time Blacklist/Watchlist Service'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Import command
    import_parser = subparsers.add_parser('import', help='Import entries from CSV')
    import_parser.add_argument('csv_file', help='Path to CSV file')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export entries to CSV')
    export_parser.add_argument('csv_file', help='Path to output CSV file')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add single entry')
    add_parser.add_argument('type', choices=['phone_number', 'device_id', 'url'],
                           help='Entry type')
    add_parser.add_argument('value', help='Value to blacklist')
    add_parser.add_argument('--reason', default='fraud', help='Reason for blacklisting')
    add_parser.add_argument('--severity', default='high',
                           choices=['low', 'medium', 'high', 'critical'],
                           help='Severity level')
    add_parser.add_argument('--source', default='manual', help='Source of entry')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove single entry')
    remove_parser.add_argument('type', choices=['phone_number', 'device_id', 'url'],
                              help='Entry type')
    remove_parser.add_argument('value', help='Value to remove')

    # Check command
    check_parser = subparsers.add_parser('check', help='Check if entry is blacklisted')
    check_parser.add_argument('type', choices=['phone_number', 'device_id', 'url'],
                             help='Entry type')
    check_parser.add_argument('value', help='Value to check')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics')

    # Rebuild command
    subparsers.add_parser('rebuild', help='Rebuild Bloom filters from database')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize service
    logger.info("Initializing Blacklist/Watchlist Service...")
    service = BlacklistWatchlistService()

    # Execute command
    if args.command == 'import':
        success = import_from_csv(service, args.csv_file)
        return 0 if success else 1

    elif args.command == 'export':
        success = export_to_csv(service, args.csv_file)
        return 0 if success else 1

    elif args.command == 'add':
        success = add_single_entry(
            service, args.type, args.value,
            args.reason, args.severity, args.source
        )
        return 0 if success else 1

    elif args.command == 'remove':
        success = remove_single_entry(service, args.type, args.value)
        return 0 if success else 1

    elif args.command == 'check':
        check_entry(service, args.type, args.value)
        return 0

    elif args.command == 'stats':
        show_statistics(service)
        return 0

    elif args.command == 'rebuild':
        rebuild_filters(service)
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
