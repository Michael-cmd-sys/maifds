"""
Flask API for MindSpore-based Phishing Detector
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('.')
from mindspore_detector import MindSporePhishingDetector
from domain_intelligence import DomainIntelligence
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize detector
logger.info("Initializing MindSpore Phishing Detector...")
detector = MindSporePhishingDetector()
logger.info(f"Detector initialized. Trained: {detector.is_trained}")

# Initialize domain intelligence
logger.info("Initializing Domain Intelligence...")
domain_intel = DomainIntelligence()
logger.info("Domain Intelligence initialized")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'framework': 'MindSpore',
        'is_trained': detector.is_trained,
        'watchlist_size': len(detector.watchlist),
        'total_detections': len(detector.detection_history)
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect phishing attempt from ad/referral signals using MindSpore neural network
    
    Expected JSON body:
    {
        "referrer_url": "http://example.com/ad",
        "ad_id": "AD_123",
        "ad_text": "Click here for free money!",
        "landing_domain_age": 10,
        "user_id": "user_456",
        "funnel_path": ["impression", "click", "form"],
        "new_recipient": false,
        "spike_in_conversions": false,
        "same_ad_used_by_many_victims": false
    }
    """
    try:
        signals = request.json
        
        if not signals:
            return jsonify({'error': 'No signals provided'}), 400
        
        # Validate required fields
        required_fields = ['referrer_url', 'ad_id', 'ad_text']
        missing_fields = [field for field in required_fields if field not in signals]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        if not detector.is_trained:
            return jsonify({
                'error': 'Model not trained. Please run training first.',
                'suggestion': 'Run: python train_mindspore.py'
            }), 503
        
        # Detect phishing using MindSpore
        detection_result = detector.detect_phishing(signals)
        
        # Take action if needed
        detector.take_action(detection_result)
        
        return jsonify(detection_result)
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    """
    Train the MindSpore detector with new data
    
    Expected JSON body:
    {
        "training_data": [
            {
                "signals": {...},
                "is_phishing": 1
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        training_data = data.get('training_data', [])
        
        if not training_data:
            return jsonify({'error': 'No training data provided'}), 400
        
        # Train the model
        detector.train(training_data, save_models=True)
        
        return jsonify({
            'status': 'success',
            'framework': 'MindSpore',
            'samples_trained': len(training_data),
            'message': 'Model trained successfully'
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/watchlist', methods=['GET', 'POST', 'DELETE'])
def manage_watchlist():
    """Manage watchlist domains"""
    if request.method == 'GET':
        # Get watchlist
        return jsonify({
            'watchlist': list(detector.watchlist),
            'size': len(detector.watchlist)
        })
    
    elif request.method == 'POST':
        # Add to watchlist
        domain = request.json.get('domain')
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        detector.add_to_watchlist(domain)
        return jsonify({
            'status': 'success',
            'domain': domain,
            'action': 'added'
        })
    
    elif request.method == 'DELETE':
        # Remove from watchlist
        domain = request.json.get('domain')
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        detector.remove_from_watchlist(domain)
        return jsonify({
            'status': 'success',
            'domain': domain,
            'action': 'removed'
        })

@app.route('/statistics', methods=['GET'])
def get_statistics():
    """Get detection statistics"""
    stats = detector.get_statistics()
    return jsonify(stats)

@app.route('/batch-detect', methods=['POST'])
def batch_detect():
    """
    Process multiple signals in batch using MindSpore
    
    Expected JSON body:
    {
        "signals_batch": [
            {...},
            {...}
        ]
    }
    """
    try:
        signals_batch = request.json.get('signals_batch', [])
        
        if not signals_batch:
            return jsonify({'error': 'No signals provided'}), 400
        
        if not detector.is_trained:
            return jsonify({
                'error': 'Model not trained',
                'suggestion': 'Run: python train_mindspore.py'
            }), 503
        
        results = []
        for signals in signals_batch:
            detection_result = detector.detect_phishing(signals)
            detector.take_action(detection_result)
            results.append(detection_result)
        
        # Summary statistics
        high_risk = sum(1 for r in results if 'HIGH_RISK' in r['risk_level'])
        medium_risk = sum(1 for r in results if 'MEDIUM_RISK' in r['risk_level'])
        low_risk = sum(1 for r in results if r['risk_level'] == 'LOW_RISK')
        
        return jsonify({
            'total_processed': len(results),
            'summary': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about the MindSpore model"""
    info = {
        'framework': 'MindSpore',
        'is_trained': detector.is_trained,
        'model_architecture': {
            'type': 'Multi-layer Perceptron (MLP)',
            'hidden_layers': detector.config['model']['hidden_sizes'],
            'dropout_rate': detector.config['model']['dropout_rate'],
            'activation': 'ReLU',
            'output': 'Softmax (Binary Classification)'
        },
        'feature_count': len(detector.feature_names) if detector.feature_names else 0,
        'training_config': {
            'learning_rate': detector.config['model']['learning_rate'],
            'epochs': detector.config['model']['epochs'],
            'batch_size': detector.config['model']['batch_size']
        }
    }
    return jsonify(info)

@app.route('/features', methods=['GET'])
def get_features():
    """Get list of features used by the model"""
    if not detector.feature_names:
        return jsonify({
            'error': 'Model not trained yet',
            'features': []
        })
    
    return jsonify({
        'feature_count': len(detector.feature_names),
        'features': detector.feature_names
    })

@app.route('/analyze-domain', methods=['POST'])
def analyze_domain():
    """
    Comprehensive domain analysis using WhoisXML API
    
    Expected JSON body:
    {
        "url": "https://example.com/suspicious-page"
    }
    
    Returns comprehensive intelligence including:
    - URL expansion (unshortening)
    - WHOIS data (domain age, owner)
    - Reputation score
    - Geolocation
    - DNS records
    - Risk assessment
    """
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL required'}), 400
        
        logger.info(f"Analyzing domain for URL: {url}")
        
        # Perform comprehensive analysis
        analysis = domain_intel.analyze_url_comprehensive(url)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Domain analysis error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/expand-url', methods=['POST'])
def expand_url():
    """
    Expand shortened URL to reveal real domain
    
    Expected JSON body:
    {
        "url": "https://bit.ly/abc123"
    }
    """
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL required'}), 400
        
        result = domain_intel.expand_url(url)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"URL expansion error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/domain-whois', methods=['POST'])
def domain_whois():
    """
    Get WHOIS data for a domain
    
    Expected JSON body:
    {
        "domain": "example.com"
    }
    """
    try:
        data = request.json
        domain = data.get('domain')
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        result = domain_intel.get_whois_data(domain)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"WHOIS lookup error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/domain-reputation', methods=['POST'])
def domain_reputation():
    """
    Check domain reputation
    
    Expected JSON body:
    {
        "domain": "example.com"
    }
    """
    try:
        data = request.json
        domain = data.get('domain')
        
        if not domain:
            return jsonify({'error': 'Domain required'}), 400
        
        result = domain_intel.check_reputation(domain)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Reputation check error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Check if model is trained
    if not detector.is_trained:
        logger.warning("⚠️  Model is not trained!")
        logger.warning("   Please run: python train_mindspore.py")
        logger.warning("   The API will start but detection will not work until trained.\n")
    else:
        logger.info("✅ Model is trained and ready!")
        logger.info(f"   Features: {len(detector.feature_names)}")
        logger.info(f"   Watchlist: {len(detector.watchlist)} domains\n")
    
    logger.info("🚀 Starting MindSpore Phishing Detector API...")
    logger.info("   API available at: http://localhost:5000")
    logger.info("   Documentation: Check endpoints with /health\n")
    
    # For production, use gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)