"""
REST API for Proactive Warning Service
Provides endpoints for campaign detection, user warnings, and statistics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

from proactive_warning_service import ProactiveWarningService

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize warning service
warning_service = None


def init_service():
    """Initialize the warning service"""
    global warning_service
    try:
        warning_service = ProactiveWarningService()
        logger.info("Proactive Warning Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Proactive Warning Service',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': warning_service.models_loaded if warning_service else False
    })


@app.route('/detect-campaign', methods=['POST'])
def detect_campaign():
    """
    Detect active scam campaign

    Request body:
    {
        "metrics": {
            "call_volume_last_hour": 150,
            "unique_targets": 80,
            "suspicious_numbers": ["+237699999999"],
            "pattern_anomaly_score": 0.92,
            "campaign_type": "call",
            "affected_users": ["user_001", "user_002"]
        }
    }

    Response:
    {
        "campaign_detected": true,
        "campaign_id": "CAMP_20251211135500",
        "risk_score": 0.92,
        "affected_users": 80,
        "actions_recommended": ["send_warnings", "enable_stricter_verification"]
    }
    """
    try:
        data = request.get_json()

        if not data or 'metrics' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Must provide "metrics" object'
            }), 400

        campaign = warning_service.detect_campaign(data['metrics'])

        if not campaign:
            return jsonify({
                'campaign_detected': False,
                'message': 'No campaign detected'
            }), 200

        return jsonify({
            'campaign_detected': True,
            'campaign_id': campaign.campaign_id,
            'campaign_type': campaign.campaign_type,
            'risk_score': campaign.risk_score,
            'volume': campaign.volume,
            'affected_users': len(campaign.affected_users),
            'suspicious_numbers': list(campaign.suspicious_numbers),
            'timestamp': campaign.start_timestamp.isoformat(),
            'actions_recommended': [
                'send_warnings',
                'enable_stricter_verification',
                'monitor_transactions'
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error in detect-campaign endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/identify-vulnerable-users', methods=['POST'])
def identify_vulnerable_users():
    """
    Identify vulnerable user cohort

    Request body:
    {
        "users": [
            {
                "user_id": "user_001",
                "recent_clicks": 3,
                "recent_calls": 5,
                "new_device": true,
                ...
            }
        ]
    }

    Response:
    {
        "vulnerable_users_count": 5,
        "vulnerable_users": [
            {
                "user_id": "user_001",
                "vulnerability_score": 0.85,
                "risk_factors": ["new_device", "high_risky_interactions"]
            }
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'users' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Must provide "users" array'
            }), 400

        vulnerable_users = warning_service.identify_vulnerable_cohort(data['users'])

        results = []
        for profile in vulnerable_users:
            results.append({
                'user_id': profile.user_id,
                'vulnerability_score': warning_service._calculate_vulnerability_score(profile),
                'risk_factors': [
                    'new_device' if profile.new_device else None,
                    'high_risky_interactions' if profile.risky_interactions_7d > 5 else None,
                    'recent_activity' if profile.recent_calls > 3 or profile.recent_clicks > 2 else None
                ]
            })

        return jsonify({
            'vulnerable_users_count': len(results),
            'vulnerable_users': results,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in identify-vulnerable-users endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/send-warnings', methods=['POST'])
def send_warnings():
    """
    Send warnings to specific users

    Request body:
    {
        "campaign_id": "CAMP_20251211135500",
        "user_ids": ["user_001", "user_002"],
        "template": "scam_warning"
    }

    Response:
    {
        "warnings_sent": 2,
        "warnings_skipped": 0,
        "notified_users": ["user_001", "user_002"]
    }
    """
    try:
        data = request.get_json()

        if not data or 'user_ids' not in data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Must provide "user_ids" array'
            }), 400

        # Get campaign if specified
        campaign = None
        if 'campaign_id' in data:
            campaign = warning_service.active_campaigns.get(data['campaign_id'])

        # Get user profiles
        vulnerable_users = []
        for user_id in data['user_ids']:
            if user_id in warning_service.user_profiles:
                vulnerable_users.append(warning_service.user_profiles[user_id])

        if not vulnerable_users:
            return jsonify({
                'error': 'No user profiles found',
                'message': 'Users must be identified first via /identify-vulnerable-users'
            }), 400

        # Send warnings
        results = warning_service.send_proactive_warnings(
            campaign or warning_service.active_campaigns[list(warning_service.active_campaigns.keys())[0]],
            vulnerable_users
        )

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error in send-warnings endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/process-campaign-alert', methods=['POST'])
def process_campaign_alert():
    """
    End-to-end campaign processing: detect, identify, warn

    Request body:
    {
        "metrics": {...},
        "users": [...]
    }

    Response:
    {
        "campaign_detected": true,
        "vulnerable_users_identified": 5,
        "warnings_sent": 3,
        "actions": [...]
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Must provide campaign data'
            }), 400

        results = warning_service.process_campaign_alert(data)

        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error in process-campaign-alert endpoint: {e}")
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
        "campaigns_detected": 10,
        "warnings_sent": 150,
        "unique_users_protected": 120,
        "active_campaigns": 2,
        "models_loaded": true
    }
    """
    try:
        stats = warning_service.get_statistics()
        return jsonify(stats), 200

    except Exception as e:
        logger.error(f"Error in statistics endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/active-campaigns', methods=['GET'])
def get_active_campaigns():
    """Get list of active campaigns"""
    try:
        campaigns = []
        for campaign_id, campaign in warning_service.active_campaigns.items():
            campaigns.append({
                'campaign_id': campaign.campaign_id,
                'campaign_type': campaign.campaign_type,
                'start_timestamp': campaign.start_timestamp.isoformat(),
                'volume': campaign.volume,
                'risk_score': campaign.risk_score,
                'affected_users_count': len(campaign.affected_users),
                'active': campaign.active
            })

        return jsonify({
            'active_campaigns_count': len(campaigns),
            'campaigns': campaigns
        }), 200

    except Exception as e:
        logger.error(f"Error in active-campaigns endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/deactivate-campaign/<campaign_id>', methods=['POST'])
def deactivate_campaign(campaign_id):
    """Deactivate a campaign"""
    try:
        if campaign_id in warning_service.active_campaigns:
            warning_service.active_campaigns[campaign_id].active = False
            return jsonify({
                'success': True,
                'message': f'Campaign {campaign_id} deactivated'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Campaign not found'
            }), 404

    except Exception as e:
        logger.error(f"Error in deactivate-campaign endpoint: {e}")
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

    parser = argparse.ArgumentParser(description='Proactive Warning Service API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5002, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Initialize service
    logger.info("Starting Proactive Warning Service API...")
    init_service()

    # Start Flask app
    logger.info(f"API server starting on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
