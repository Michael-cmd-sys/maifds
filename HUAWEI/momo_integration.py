from mindspore_detector import MindSporePhishingDetector

# Initialize once at app startup
detector = MindSporePhishingDetector()

def process_momo_transaction(transaction):
    """
    Call this function for every MoMo transaction
    """
    # Extract signals from your transaction
    signals = {
        'referrer_url': transaction.get('referrer_url', ''),
        'ad_id': transaction.get('ad_id', ''),
        'ad_text': transaction.get('ad_metadata', {}).get('text', ''),
        'landing_domain_age': None,  # Will fetch via WHOIS
        'user_id': transaction['user_id'],
        'funnel_path': transaction.get('user_journey', []),
        'new_recipient': is_new_recipient(transaction),
        'spike_in_conversions': detect_spike(transaction),
        'same_ad_used_by_many_victims': check_ad_pattern(transaction)
    }
    
    # Detect phishing
    result = detector.detect_phishing(signals)
    
    # Take action based on risk
    if result['risk_level'].startswith('HIGH_RISK'):
        # Block transaction
        block_transaction(transaction['id'])
        send_user_alert(transaction['user_id'])
        log_fraud_attempt(transaction, result)
        return {"allowed": False, "reason": "High fraud risk"}
    
    elif result['risk_level'].startswith('MEDIUM_RISK'):
        # Allow but flag for review
        flag_for_manual_review(transaction['id'])
        return {"allowed": True, "flagged": True}
    
    else:
        # Allow normally
        return {"allowed": True}