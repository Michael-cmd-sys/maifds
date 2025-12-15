"""
Real-time Blacklist/Watchlist Service with Bloom Filters
Ultra-fast lookup of known bad numbers/URLs/devices using in-memory Bloom filters
"""

import hashlib
import json
import logging
import pickle
import os
import redis
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from bitarray import bitarray
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blacklist_watchlist_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Bloom filter for ultra-fast membership testing
    Space-efficient probabilistic data structure for set membership
    """

    def __init__(self, expected_elements: int = 1000000, false_positive_rate: float = 0.01):
        """
        Initialize Bloom filter

        Args:
            expected_elements: Expected number of elements to store
            false_positive_rate: Desired false positive rate (0-1)
        """
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        # Calculate optimal size and number of hash functions
        self.size = self._calculate_size(expected_elements, false_positive_rate)
        self.num_hashes = self._calculate_num_hashes(self.size, expected_elements)

        # Initialize bit array
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

        # Statistics
        self.num_elements = 0

        logger.info(f"Bloom filter initialized: size={self.size} bits, num_hashes={self.num_hashes}")

    @staticmethod
    def _calculate_size(n: int, p: float) -> int:
        """Calculate optimal bit array size"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _calculate_num_hashes(m: int, n: int) -> int:
        """Calculate optimal number of hash functions"""
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hash(self, item: str, seed: int) -> int:
        """Generate hash with seed"""
        hash_obj = hashlib.sha256(f"{item}{seed}".encode())
        hash_value = int(hash_obj.hexdigest(), 16)
        return hash_value % self.size

    def add(self, item: str):
        """Add item to Bloom filter"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.num_elements += 1

    def contains(self, item: str) -> bool:
        """Check if item might be in the set (O(1) operation)"""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True

    def batch_add(self, items: List[str]):
        """Add multiple items efficiently"""
        for item in items:
            self.add(item)
        logger.info(f"Added {len(items)} items to Bloom filter")

    def get_stats(self) -> Dict:
        """Get Bloom filter statistics"""
        bits_set = self.bit_array.count(1)
        load_factor = bits_set / self.size

        # Estimate actual false positive rate
        actual_fpr = (1 - math.exp(-self.num_hashes * self.num_elements / self.size)) ** self.num_hashes

        return {
            'size_bits': self.size,
            'size_mb': self.size / (8 * 1024 * 1024),
            'num_hashes': self.num_hashes,
            'num_elements': self.num_elements,
            'bits_set': bits_set,
            'load_factor': load_factor,
            'expected_fpr': self.false_positive_rate,
            'actual_fpr': actual_fpr
        }

    def save(self, filepath: str):
        """Save Bloom filter to disk"""
        data = {
            'expected_elements': self.expected_elements,
            'false_positive_rate': self.false_positive_rate,
            'size': self.size,
            'num_hashes': self.num_hashes,
            'num_elements': self.num_elements,
            'bit_array': self.bit_array.tobytes()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Bloom filter saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load Bloom filter from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        bf = cls(data['expected_elements'], data['false_positive_rate'])
        bf.size = data['size']
        bf.num_hashes = data['num_hashes']
        bf.num_elements = data['num_elements']
        bf.bit_array = bitarray()
        bf.bit_array.frombytes(data['bit_array'])

        logger.info(f"Bloom filter loaded from {filepath}")
        return bf


class HashGenerator:
    """Generate secure hashes for phone numbers, devices, and URLs"""

    @staticmethod
    def hash_phone_number(phone_number: str, salt: str = "") -> str:
        """
        Hash phone number with optional salt
        Supports international format normalization
        """
        # Normalize: remove spaces, dashes, parentheses
        normalized = ''.join(c for c in phone_number if c.isdigit() or c == '+')

        # Hash with SHA-256
        hash_input = f"{normalized}{salt}".encode()
        hash_obj = hashlib.sha256(hash_input)
        return hash_obj.hexdigest()

    @staticmethod
    def hash_device_id(device_id: str, salt: str = "") -> str:
        """Hash device ID (IMEI, MAC address, etc.)"""
        normalized = device_id.strip().lower()
        hash_input = f"{normalized}{salt}".encode()
        hash_obj = hashlib.sha256(hash_input)
        return hash_obj.hexdigest()

    @staticmethod
    def hash_url(url: str, salt: str = "") -> str:
        """Hash URL with normalization"""
        # Normalize URL: lowercase, remove trailing slash
        normalized = url.strip().lower().rstrip('/')
        hash_input = f"{normalized}{salt}".encode()
        hash_obj = hashlib.sha256(hash_input)
        return hash_obj.hexdigest()

    @staticmethod
    def hash_generic(value: str, salt: str = "") -> str:
        """Generic hash function"""
        hash_input = f"{value}{salt}".encode()
        hash_obj = hashlib.sha256(hash_input)
        return hash_obj.hexdigest()


class BlacklistDatabase:
    """
    Authoritative database for blacklisted entries
    Used for verification after Bloom filter positive
    """

    def __init__(self, db_path: str = "data/blacklist_db.json"):
        self.db_path = db_path
        self.data = {
            'phone_numbers': {},
            'device_ids': {},
            'urls': {},
            'metadata': {
                'last_updated': None,
                'total_entries': 0
            }
        }
        self._load_database()

    def _load_database(self):
        """Load database from file"""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    self.data = json.load(f)
                logger.info(f"Blacklist database loaded: {self.data['metadata']['total_entries']} entries")
            except Exception as e:
                logger.error(f"Error loading database: {e}")

    def _save_database(self):
        """Save database to file"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.data['metadata']['last_updated'] = datetime.now().isoformat()
        self.data['metadata']['total_entries'] = (
            len(self.data['phone_numbers']) +
            len(self.data['device_ids']) +
            len(self.data['urls'])
        )

        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2)
            logger.info(f"Database saved: {self.data['metadata']['total_entries']} entries")
        except Exception as e:
            logger.error(f"Error saving database: {e}")

    def add_entry(self, entry_type: str, hash_value: str, metadata: Dict):
        """Add entry to blacklist"""
        if entry_type not in self.data:
            self.data[entry_type] = {}

        self.data[entry_type][hash_value] = {
            'added_date': datetime.now().isoformat(),
            'reason': metadata.get('reason', 'fraud'),
            'severity': metadata.get('severity', 'high'),
            'source': metadata.get('source', 'manual'),
            'additional_info': metadata.get('additional_info', {})
        }
        self._save_database()

    def get_entry(self, entry_type: str, hash_value: str) -> Optional[Dict]:
        """Get entry details"""
        return self.data.get(entry_type, {}).get(hash_value)

    def remove_entry(self, entry_type: str, hash_value: str) -> bool:
        """Remove entry from blacklist"""
        if entry_type in self.data and hash_value in self.data[entry_type]:
            del self.data[entry_type][hash_value]
            self._save_database()
            return True
        return False

    def batch_add(self, entries: List[Tuple[str, str, Dict]]):
        """Add multiple entries efficiently"""
        for entry_type, hash_value, metadata in entries:
            if entry_type not in self.data:
                self.data[entry_type] = {}

            self.data[entry_type][hash_value] = {
                'added_date': datetime.now().isoformat(),
                'reason': metadata.get('reason', 'fraud'),
                'severity': metadata.get('severity', 'high'),
                'source': metadata.get('source', 'manual'),
                'additional_info': metadata.get('additional_info', {})
            }

        self._save_database()
        logger.info(f"Batch added {len(entries)} entries")


class RedisCache:
    """Redis cache for ultra-fast O(1) lookups"""

    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Running without cache.")
            self.redis_client = None
            self.enabled = False

    def set_blacklisted(self, key: str, value: Dict, expiry: int = 86400):
        """Set blacklist entry in Redis with expiry (default 24h)"""
        if not self.enabled:
            return

        try:
            self.redis_client.setex(
                f"blacklist:{key}",
                expiry,
                json.dumps(value)
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")

    def get_blacklisted(self, key: str) -> Optional[Dict]:
        """Get blacklist entry from Redis"""
        if not self.enabled:
            return None

        try:
            value = self.redis_client.get(f"blacklist:{key}")
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None

    def delete_blacklisted(self, key: str):
        """Remove entry from Redis cache"""
        if not self.enabled:
            return

        try:
            self.redis_client.delete(f"blacklist:{key}")
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")


class BlacklistWatchlistService:
    """
    Main Real-time Blacklist/Watchlist Service
    Uses Bloom filters for O(1) membership testing + Redis cache + authoritative DB
    """

    def __init__(self, config_path: str = "config_blacklist.json"):
        """Initialize the blacklist service"""
        self.config = self._load_config(config_path)

        # Initialize components
        self.hash_generator = HashGenerator()
        self.bloom_filter_numbers = None
        self.bloom_filter_devices = None
        self.bloom_filter_urls = None
        self.database = BlacklistDatabase(self.config['storage']['database_path'])
        self.cache = RedisCache(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db']
        )

        # Statistics
        self.detection_stats = {
            'total_checks': 0,
            'total_hits': 0,
            'phone_hits': 0,
            'device_hits': 0,
            'url_hits': 0,
            'blocked_transactions': 0
        }

        # Initialize Bloom filters
        self._initialize_bloom_filters()

        logger.info("Blacklist/Watchlist Service initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
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
                "bloom_filters_dir": "data/bloom_filters/",
                "checkpoint_prefix": "blacklist"
            },
            "actions": {
                "auto_block": True,
                "enforce_high_verification": True,
                "send_notifications": True
            },
            "notifications": {
                "webhook_url": "https://your-webhook-url.com/fraud-alert",
                "sms_enabled": True,
                "email_enabled": True
            },
            "thresholds": {
                "max_severity_for_auto_block": "high"
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")

        return default_config

    def _initialize_bloom_filters(self):
        """Initialize or load Bloom filters"""
        bloom_dir = self.config['storage']['bloom_filters_dir']
        os.makedirs(bloom_dir, exist_ok=True)

        expected_elements = self.config['bloom_filter']['expected_elements']
        fpr = self.config['bloom_filter']['false_positive_rate']

        # Phone numbers Bloom filter
        numbers_path = os.path.join(bloom_dir, 'numbers.bloom')
        if os.path.exists(numbers_path):
            self.bloom_filter_numbers = BloomFilter.load(numbers_path)
        else:
            self.bloom_filter_numbers = BloomFilter(expected_elements, fpr)

        # Device IDs Bloom filter
        devices_path = os.path.join(bloom_dir, 'devices.bloom')
        if os.path.exists(devices_path):
            self.bloom_filter_devices = BloomFilter.load(devices_path)
        else:
            self.bloom_filter_devices = BloomFilter(expected_elements, fpr)

        # URLs Bloom filter
        urls_path = os.path.join(bloom_dir, 'urls.bloom')
        if os.path.exists(urls_path):
            self.bloom_filter_urls = BloomFilter.load(urls_path)
        else:
            self.bloom_filter_urls = BloomFilter(expected_elements, fpr)

    def add_to_blacklist(self, entry_type: str, value: str, metadata: Dict) -> Dict:
        """
        Add entry to blacklist (Bloom filter + DB + cache)

        Args:
            entry_type: 'phone_number', 'device_id', or 'url'
            value: The actual value to blacklist
            metadata: Additional information (reason, severity, source, etc.)

        Returns:
            Result dictionary
        """
        # Generate hash
        if entry_type == 'phone_number':
            hash_value = self.hash_generator.hash_phone_number(value)
            bloom_filter = self.bloom_filter_numbers
        elif entry_type == 'device_id':
            hash_value = self.hash_generator.hash_device_id(value)
            bloom_filter = self.bloom_filter_devices
        elif entry_type == 'url':
            hash_value = self.hash_generator.hash_url(value)
            bloom_filter = self.bloom_filter_urls
        else:
            return {'success': False, 'error': f'Invalid entry type: {entry_type}'}

        # Add to Bloom filter
        bloom_filter.add(hash_value)

        # Add to database
        self.database.add_entry(entry_type, hash_value, metadata)

        # Add to Redis cache
        cache_data = {
            'hash': hash_value,
            'type': entry_type,
            'metadata': metadata
        }
        self.cache.set_blacklisted(
            hash_value,
            cache_data,
            self.config['redis']['cache_expiry']
        )

        logger.info(f"Added to blacklist: {entry_type} - {hash_value[:16]}...")

        return {
            'success': True,
            'hash': hash_value,
            'type': entry_type,
            'message': 'Entry added to blacklist'
        }

    def check_blacklist(self, signals: Dict) -> Dict:
        """
        Check if any signal matches blacklist
        Ultra-fast O(1) check using Bloom filter

        Args:
            signals: Dictionary with 'phone_number', 'device_id', 'url'

        Returns:
            Detection result with actions
        """
        self.detection_stats['total_checks'] += 1

        results = {
            'is_blacklisted': False,
            'matches': [],
            'severity': 'none',
            'actions': [],
            'details': {}
        }

        # Check phone number
        if 'phone_number' in signals:
            phone_hash = self.hash_generator.hash_phone_number(signals['phone_number'])
            if self._check_entry('phone_number', phone_hash, self.bloom_filter_numbers):
                results['is_blacklisted'] = True
                results['matches'].append('phone_number')
                self.detection_stats['phone_hits'] += 1

        # Check device ID
        if 'device_id' in signals:
            device_hash = self.hash_generator.hash_device_id(signals['device_id'])
            if self._check_entry('device_id', device_hash, self.bloom_filter_devices):
                results['is_blacklisted'] = True
                results['matches'].append('device_id')
                self.detection_stats['device_hits'] += 1

        # Check URL
        if 'url' in signals:
            url_hash = self.hash_generator.hash_url(signals['url'])
            if self._check_entry('url', url_hash, self.bloom_filter_urls):
                results['is_blacklisted'] = True
                results['matches'].append('url')
                self.detection_stats['url_hits'] += 1

        # Determine actions if blacklisted
        if results['is_blacklisted']:
            self.detection_stats['total_hits'] += 1
            results['severity'] = 'high'
            results['actions'] = self._determine_actions(results)

            # Log detection
            logger.warning(f"BLACKLIST HIT: {results['matches']} - Signals: {signals}")

        return results

    def _check_entry(self, entry_type: str, hash_value: str, bloom_filter: BloomFilter) -> bool:
        """
        Check if entry is blacklisted
        1. Check Redis cache (O(1))
        2. Check Bloom filter (O(1))
        3. If Bloom filter positive, verify with authoritative DB
        """
        # Step 1: Check Redis cache
        cached = self.cache.get_blacklisted(hash_value)
        if cached:
            logger.debug(f"Cache hit: {entry_type}")
            return True

        # Step 2: Check Bloom filter
        if not bloom_filter.contains(hash_value):
            # Definitely not in blacklist
            return False

        # Step 3: Bloom filter says "possibly in set" - verify with DB
        db_entry = self.database.get_entry(entry_type, hash_value)
        if db_entry:
            # True positive - update cache
            self.cache.set_blacklisted(
                hash_value,
                {'type': entry_type, 'metadata': db_entry},
                self.config['redis']['cache_expiry']
            )
            return True

        # False positive from Bloom filter
        logger.debug(f"Bloom filter false positive: {entry_type}")
        return False

    def _determine_actions(self, results: Dict) -> List[str]:
        """Determine actions to take based on detection"""
        actions = []

        config_actions = self.config['actions']

        if config_actions['auto_block']:
            actions.append('BLOCK_TRANSACTION')
            self.detection_stats['blocked_transactions'] += 1

        if config_actions['enforce_high_verification']:
            actions.append('ENFORCE_HIGH_VERIFICATION')

        if config_actions['send_notifications']:
            actions.append('SEND_NOTIFICATION_TO_FRAUD_OPS')
            actions.append('SEND_NOTIFICATION_TO_USER')

        return actions

    def batch_add_to_blacklist(self, entries: List[Tuple[str, str, Dict]]):
        """Add multiple entries efficiently"""
        for entry_type, value, metadata in entries:
            self.add_to_blacklist(entry_type, value, metadata)

        # Save Bloom filters after batch update
        self.save_bloom_filters()

        logger.info(f"Batch added {len(entries)} entries to blacklist")

    def remove_from_blacklist(self, entry_type: str, value: str) -> Dict:
        """Remove entry from blacklist"""
        # Generate hash
        if entry_type == 'phone_number':
            hash_value = self.hash_generator.hash_phone_number(value)
        elif entry_type == 'device_id':
            hash_value = self.hash_generator.hash_device_id(value)
        elif entry_type == 'url':
            hash_value = self.hash_generator.hash_url(value)
        else:
            return {'success': False, 'error': f'Invalid entry type: {entry_type}'}

        # Remove from database
        success = self.database.remove_entry(entry_type, hash_value)

        # Remove from cache
        self.cache.delete_blacklisted(hash_value)

        # Note: Cannot remove from Bloom filter (limitation of Bloom filters)
        # Need to rebuild Bloom filter if many removals

        if success:
            logger.info(f"Removed from blacklist: {entry_type} - {hash_value[:16]}...")
            return {'success': True, 'message': 'Entry removed from blacklist'}
        else:
            return {'success': False, 'error': 'Entry not found in blacklist'}

    def save_bloom_filters(self):
        """Save all Bloom filters to disk"""
        bloom_dir = self.config['storage']['bloom_filters_dir']
        os.makedirs(bloom_dir, exist_ok=True)

        self.bloom_filter_numbers.save(os.path.join(bloom_dir, 'numbers.bloom'))
        self.bloom_filter_devices.save(os.path.join(bloom_dir, 'devices.bloom'))
        self.bloom_filter_urls.save(os.path.join(bloom_dir, 'urls.bloom'))

        logger.info("All Bloom filters saved")

    def get_statistics(self) -> Dict:
        """Get service statistics"""
        return {
            'detection_stats': self.detection_stats,
            'bloom_filter_numbers': self.bloom_filter_numbers.get_stats(),
            'bloom_filter_devices': self.bloom_filter_devices.get_stats(),
            'bloom_filter_urls': self.bloom_filter_urls.get_stats(),
            'database_entries': self.database.data['metadata']['total_entries'],
            'redis_cache_enabled': self.cache.enabled
        }

    def rebuild_bloom_filters(self):
        """
        Rebuild Bloom filters from authoritative database
        Use this after many deletions
        """
        logger.info("Rebuilding Bloom filters from database...")

        # Create new Bloom filters
        expected_elements = self.config['bloom_filter']['expected_elements']
        fpr = self.config['bloom_filter']['false_positive_rate']

        self.bloom_filter_numbers = BloomFilter(expected_elements, fpr)
        self.bloom_filter_devices = BloomFilter(expected_elements, fpr)
        self.bloom_filter_urls = BloomFilter(expected_elements, fpr)

        # Repopulate from database
        for hash_value in self.database.data.get('phone_numbers', {}):
            self.bloom_filter_numbers.add(hash_value)

        for hash_value in self.database.data.get('device_ids', {}):
            self.bloom_filter_devices.add(hash_value)

        for hash_value in self.database.data.get('urls', {}):
            self.bloom_filter_urls.add(hash_value)

        # Save rebuilt filters
        self.save_bloom_filters()

        logger.info("Bloom filters rebuilt successfully")


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = BlacklistWatchlistService()

    # Add entries to blacklist
    service.add_to_blacklist(
        'phone_number',
        '+1234567890',
        {
            'reason': 'fraud',
            'severity': 'high',
            'source': 'fraud_ops',
            'additional_info': {'reported_by': 'user_complaints', 'num_reports': 5}
        }
    )

    service.add_to_blacklist(
        'device_id',
        'IMEI-123456789012345',
        {
            'reason': 'stolen_device',
            'severity': 'critical',
            'source': 'telco'
        }
    )

    service.add_to_blacklist(
        'url',
        'https://phishing-site.xyz/fake-login',
        {
            'reason': 'phishing',
            'severity': 'high',
            'source': 'security_scan'
        }
    )

    # Check signals
    test_signals = {
        'phone_number': '+1234567890',
        'device_id': 'IMEI-123456789012345',
        'url': 'https://phishing-site.xyz/fake-login'
    }

    result = service.check_blacklist(test_signals)
    print(json.dumps(result, indent=2))

    # Get statistics
    stats = service.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))

    # Save Bloom filters
    service.save_bloom_filters()
