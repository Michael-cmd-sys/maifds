"""
REST API for Real-time Blacklist/Watchlist Service
Provides endpoints for checking, adding, and managing blacklist entries
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from datetime import datetime
from pydantic import BaseModel
from typing import Optional


from blacklist_watchlist_service import BlacklistWatchlistService

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize blacklist service
blacklist_service = None

# Pydantic models for request validation
class BlacklistCheckRequest(BaseModel):
    phone_number: Optional[str] = None
    device_id: Optional[str] = None
    url: Optional[str] = None


def init_service():
    """Initialize the blacklist service"""
    global blacklist_service
    try:
        blacklist_service = BlacklistWatchlistService()
        logger.info("Blacklist/Watchlist Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Blacklist/Watchlist Service',
        'timestamp': datetime.now().isoformat(),
        'redis_enabled': blacklist_service.cache.enabled if blacklist_service else False
    })


@app.route('/check', methods=['POST'])
def check_blacklist():
    """
    Check if phone number, device ID, or URL is blacklisted

    Request body:
    {
        "phone_number": "+1234567890",  // optional
        "device_id": "IMEI-123456",     // optional
        "url": "https://example.com",   // optional
        "transaction_id": "TXN_123"     // optional, for tracking
    }

    Response:
    {
        "is_blacklisted": true,
        "matches": ["phone_number", "url"],
        "severity": "high",
        "actions": ["BLOCK_TRANSACTION", "SEND_NOTIFICATION"],
        "timestamp": "2025-12-11T12:00:00",
        "transaction_id": "TXN_123"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must be JSON'
            }), 400

        # Extract signals
        signals = {}
        if 'phone_number' in data:
            signals['phone_number'] = data['phone_number']
        if 'device_id' in data:
            signals['device_id'] = data['device_id']
        if 'url' in data:
            signals['url'] = data['url']

        if not signals:
            return jsonify({
                'error': 'No signals provided',
                'message': 'Must provide at least one of: phone_number, device_id, url'
            }), 400

        # Check blacklist
        result = blacklist_service.check_blacklist(signals)

        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        if 'transaction_id' in data:
            result['transaction_id'] = data['transaction_id']

        # Log check
        logger.info(f"Check: {signals} - Blacklisted: {result['is_blacklisted']}")

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in check endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/batch-check', methods=['POST'])
def batch_check_blacklist():
    """
    Check multiple entries at once

    Request body:
    {
        "checks": [
            {"phone_number": "+1234567890"},
            {"device_id": "IMEI-123456"},
            {"url": "https://example.com"}
        ]
    }

    Response:
    {
        "results": [
            {"index": 0, "is_blacklisted": true, ...},
            {"index": 1, "is_blacklisted": false, ...},
            ...
        ],
        "summary": {
            "total": 3,
            "blacklisted": 1,
            "clean": 2
        }
    }
    """
    try:
        data = request.get_json()

        if not data or 'checks' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request must contain "checks" array'
            }), 400

        checks = data['checks']
        results = []
        blacklisted_count = 0

        for idx, signals in enumerate(checks):
            result = blacklist_service.check_blacklist(signals)
            result['index'] = idx
            results.append(result)

            if result['is_blacklisted']:
                blacklisted_count += 1

        summary = {
            'total': len(checks),
            'blacklisted': blacklisted_count,
            'clean': len(checks) - blacklisted_count
        }

        return jsonify({
            'results': results,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in batch-check endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/add', methods=['POST'])
def add_to_blacklist():
    """
    Add entry to blacklist

    Request body:
    {
        "type": "phone_number",  // phone_number, device_id, or url
        "value": "+1234567890",
        "reason": "fraud",
        "severity": "high",      // low, medium, high, critical
        "source": "manual",
        "additional_info": {}    // optional
    }

    Response:
    {
        "success": true,
        "hash": "abc123...",
        "type": "phone_number",
        "message": "Entry added to blacklist"
    }
    """
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['type', 'value', 'reason', 'severity', 'source']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'required_fields': required_fields
                }), 400

        # Validate type
        valid_types = ['phone_number', 'device_id', 'url']
        if data['type'] not in valid_types:
            return jsonify({
                'error': f'Invalid type: {data["type"]}',
                'valid_types': valid_types
            }), 400

        # Validate severity
        valid_severities = ['low', 'medium', 'high', 'critical']
        if data['severity'] not in valid_severities:
            return jsonify({
                'error': f'Invalid severity: {data["severity"]}',
                'valid_severities': valid_severities
            }), 400

        # Prepare metadata
        metadata = {
            'reason': data['reason'],
            'severity': data['severity'],
            'source': data['source'],
            'additional_info': data.get('additional_info', {})
        }

        # Add to blacklist
        result = blacklist_service.add_to_blacklist(
            data['type'],
            data['value'],
            metadata
        )

        # Save Bloom filters
        blacklist_service.save_bloom_filters()

        logger.info(f"Added to blacklist: {data['type']} - {data['value']}")

        return jsonify(result), 201 if result['success'] else 400

    except Exception as e:
        logger.error(f"Error in add endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/batch-add', methods=['POST'])
def batch_add_to_blacklist():
    """
    Add multiple entries to blacklist

    Request body:
    {
        "entries": [
            {
                "type": "phone_number",
                "value": "+1234567890",
                "reason": "fraud",
                "severity": "high",
                "source": "manual"
            },
            ...
        ]
    }

    Response:
    {
        "success": true,
        "added": 10,
        "message": "Batch added 10 entries"
    }
    """
    try:
        data = request.get_json()

        if not data or 'entries' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request must contain "entries" array'
            }), 400

        entries_data = data['entries']
        entries = []

        for entry_data in entries_data:
            metadata = {
                'reason': entry_data.get('reason', 'fraud'),
                'severity': entry_data.get('severity', 'high'),
                'source': entry_data.get('source', 'manual'),
                'additional_info': entry_data.get('additional_info', {})
            }
            entries.append((entry_data['type'], entry_data['value'], metadata))

        # Batch add
        blacklist_service.batch_add_to_blacklist(entries)

        logger.info(f"Batch added {len(entries)} entries")

        return jsonify({
            'success': True,
            'added': len(entries),
            'message': f'Batch added {len(entries)} entries'
        }), 201

    except Exception as e:
        logger.error(f"Error in batch-add endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/remove', methods=['POST'])
def remove_from_blacklist():
    """
    Remove entry from blacklist

    Request body:
    {
        "type": "phone_number",
        "value": "+1234567890"
    }

    Response:
    {
        "success": true,
        "message": "Entry removed from blacklist"
    }
    """
    try:
        data = request.get_json()

        if not data or 'type' not in data or 'value' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Must provide "type" and "value"'
            }), 400

        result = blacklist_service.remove_from_blacklist(data['type'], data['value'])

        logger.info(f"Removed from blacklist: {data['type']} - {data['value']}")

        return jsonify(result), 200 if result['success'] else 400

    except Exception as e:
        logger.error(f"Error in remove endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/statistics', methods=['GET'])
def get_statistics():
    """
    Get service statistics

    Response:
    {
        "detection_stats": {...},
        "bloom_filter_numbers": {...},
        "bloom_filter_devices": {...},
        "bloom_filter_urls": {...},
        "database_entries": 1000,
        "redis_cache_enabled": true
    }
    """
    try:
        stats = blacklist_service.get_statistics()
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error in statistics endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/rebuild', methods=['POST'])
def rebuild_bloom_filters():
    """
    Rebuild Bloom filters from database
    Use this after many deletions

    Response:
    {
        "success": true,
        "message": "Bloom filters rebuilt successfully"
    }
    """
    try:
        blacklist_service.rebuild_bloom_filters()

        return jsonify({
            'success': True,
            'message': 'Bloom filters rebuilt successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error in rebuild endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Blacklist/Watchlist Service API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Initialize service
    logger.info("Starting Blacklist/Watchlist Service API...")
    init_service()

    # Start Flask app
    logger.info(f"API server starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
