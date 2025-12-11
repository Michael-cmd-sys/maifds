"""
Training script for MindSpore-based Phishing Detector
"""

import sys
sys.path.append('.')
from mindspore_detector import MindSporePhishingDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_training_data():
    """Generate synthetic training data"""
    
    # Legitimate ads
    legit_ads = [
        {
            'signals': {
                'referrer_url': 'https://www.legitimate-store.com/promo',
                'ad_id': 'L_001',
                'landing_domain_age': 1000,
                'ad_text': 'New product launch - Check out our latest electronics collection',
                'funnel_path': ['impression', 'click', 'view'],
                'user_id': 'user_001',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://offers.trusted-brand.net/summer-sale',
                'ad_id': 'L_002',
                'landing_domain_age': 2000,
                'ad_text': 'Amazing deals on all electronics - Shop now and save up to 50%',
                'funnel_path': ['impression', 'click', 'add_cart', 'purchase'],
                'user_id': 'user_002',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://secure.online-shop.org/deals',
                'ad_id': 'L_003',
                'landing_domain_age': 1500,
                'ad_text': 'Quality products at great prices - Browse our catalog',
                'funnel_path': ['impression', 'click', 'view'],
                'user_id': 'user_003',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://www.official-retailer.com/promotions',
                'ad_id': 'L_004',
                'landing_domain_age': 3000,
                'ad_text': 'Official retailer - Authorized dealer for top brands',
                'funnel_path': ['impression', 'click', 'product_view', 'purchase'],
                'user_id': 'user_004',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://shopping.trusted-merchant.co/flash-sale',
                'ad_id': 'L_005',
                'landing_domain_age': 1200,
                'ad_text': 'Flash sale this weekend only - Limited time offer',
                'funnel_path': ['impression', 'click', 'view'],
                'user_id': 'user_005',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://deals.major-retailer.com/clearance',
                'ad_id': 'L_006',
                'landing_domain_age': 2500,
                'ad_text': 'Clearance sale - Up to 70% off selected items',
                'funnel_path': ['impression', 'click', 'view', 'purchase'],
                'user_id': 'user_006',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://www.electronics-store.co/seasonal',
                'ad_id': 'L_007',
                'landing_domain_age': 1800,
                'ad_text': 'Seasonal promotion on home electronics and appliances',
                'funnel_path': ['impression', 'click', 'view'],
                'user_id': 'user_007',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        },
        {
            'signals': {
                'referrer_url': 'https://shop.reliable-vendor.net/offers',
                'ad_id': 'L_008',
                'landing_domain_age': 2200,
                'ad_text': 'Great offers on quality products - Check our new arrivals',
                'funnel_path': ['impression', 'click', 'view', 'add_cart'],
                'user_id': 'user_008',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            },
            'is_phishing': 0
        }
    ]
    
    # Phishing ads
    phishing_ads = [
        {
            'signals': {
                'referrer_url': 'http://bank-security-verify.xyz/urgent',
                'ad_id': 'P_001',
                'landing_domain_age': 5,
                'ad_text': 'URGENT: Your bank account has been suspended! Click here to verify now!!!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_101',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://free-money-claim.gq/winner',
                'ad_id': 'P_002',
                'landing_domain_age': 10,
                'ad_text': 'Congratulations! You won $10,000! Claim your prize now by clicking here!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_102',
                'new_recipient': True,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://momo-security-alert.tk/verify',
                'ad_id': 'P_003',
                'landing_domain_age': 3,
                'ad_text': 'Warning! Your MoMo account will be blocked in 24 hours. Verify your identity immediately!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_103',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://system-repair-download.cn/fix',
                'ad_id': 'P_004',
                'landing_domain_age': 7,
                'ad_text': 'Your phone is infected with 5 viruses! Download our antivirus app now!!!',
                'funnel_path': ['impression', 'click', 'download'],
                'user_id': 'user_104',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://urgent-tax-refund.ml/claim',
                'ad_id': 'P_005',
                'landing_domain_age': 15,
                'ad_text': 'Urgent: You have an unclaimed tax refund of $5,000. Click to claim before it expires!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_105',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://mobile-money-bonus.ga/activate',
                'ad_id': 'P_006',
                'landing_domain_age': 2,
                'ad_text': 'Activate your mobile money bonus now! Enter your PIN to receive 1000 free credits!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_106',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://customer-service-verification.info/update',
                'ad_id': 'P_007',
                'landing_domain_age': 12,
                'ad_text': 'Important security update required. Verify your account details to continue using services.',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_107',
                'new_recipient': True,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://account-suspended-alert.cf/restore',
                'ad_id': 'P_008',
                'landing_domain_age': 8,
                'ad_text': 'ALERT: Account suspended due to suspicious activity! Click here immediately to restore access!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_108',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://win-iphone-free.tk/claim',
                'ad_id': 'P_009',
                'landing_domain_age': 4,
                'ad_text': 'You are the 1000th visitor! Win a FREE iPhone! Click to claim your prize now!!!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_109',
                'new_recipient': True,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        },
        {
            'signals': {
                'referrer_url': 'http://payment-verification-urgent.gq/secure',
                'ad_id': 'P_010',
                'landing_domain_age': 6,
                'ad_text': 'Urgent payment verification required! Your payment is on hold. Verify now to release funds!',
                'funnel_path': ['impression', 'click', 'form_submission'],
                'user_id': 'user_110',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': True
            },
            'is_phishing': 1
        }
    ]
    
    return legit_ads + phishing_ads

def main():
    """Main training function"""
    logger.info("=" * 70)
    logger.info("MindSpore Phishing Detector - Training Script")
    logger.info("=" * 70)
    
    # Initialize detector
    logger.info("\nInitializing MindSpore detector...")
    detector = MindSporePhishingDetector(config_path="config.json")
    
    # Generate training data
    logger.info("\nGenerating synthetic training data...")
    training_data = generate_training_data()
    logger.info(f"Generated {len(training_data)} training samples")
    
    # Count samples
    num_phishing = sum(1 for item in training_data if item['is_phishing'] == 1)
    num_legit = len(training_data) - num_phishing
    logger.info(f"  - Legitimate ads: {num_legit}")
    logger.info(f"  - Phishing ads: {num_phishing}")
    
    # Train the model
    logger.info("\n" + "=" * 70)
    logger.info("Training MindSpore Neural Network...")
    logger.info("=" * 70)
    
    try:
        detector.train(training_data, save_models=True)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Training Complete!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test on sample data
    logger.info("\n" + "=" * 70)
    logger.info("Testing on Sample Data...")
    logger.info("=" * 70)
    
    test_signals = [
        {
            'name': 'High-Risk Phishing',
            'signals': {
                'referrer_url': 'http://suspicious-bank-verify.xyz/login',
                'ad_id': 'TEST_001',
                'landing_domain_age': 2,
                'ad_text': 'URGENT! Your account will be closed! Verify now!',
                'funnel_path': ['impression', 'click', 'form'],
                'user_id': 'test_user_1',
                'new_recipient': True,
                'spike_in_conversions': True,
                'same_ad_used_by_many_victims': False
            }
        },
        {
            'name': 'Legitimate Ad',
            'signals': {
                'referrer_url': 'https://www.legitimate-shop.com/sale',
                'ad_id': 'TEST_002',
                'landing_domain_age': 2000,
                'ad_text': 'Check out our new collection of products',
                'funnel_path': ['impression', 'click', 'view'],
                'user_id': 'test_user_2',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            }
        },
        {
            'name': 'Medium-Risk Suspicious',
            'signals': {
                'referrer_url': 'http://new-deals-site.com/offers',
                'ad_id': 'TEST_003',
                'landing_domain_age': 45,
                'ad_text': 'Amazing offers! Limited time only!',
                'funnel_path': ['impression', 'click'],
                'user_id': 'test_user_3',
                'new_recipient': False,
                'spike_in_conversions': False,
                'same_ad_used_by_many_victims': False
            }
        }
    ]
    
    for i, test_case in enumerate(test_signals, 1):
        logger.info(f"\n{'─' * 70}")
        logger.info(f"Test Case {i}: {test_case['name']}")
        logger.info(f"{'─' * 70}")
        
        result = detector.detect_phishing(test_case['signals'])
        
        logger.info(f"📊 Results:")
        logger.info(f"   Risk Level: {result['risk_level']}")
        logger.info(f"   Confidence: {result['confidence']:.2%}")
        if result['reasons']:
            logger.info(f"   Reasons:")
            for reason in result['reasons']:
                logger.info(f"      • {reason}")
    
    # Display statistics
    logger.info("\n" + "=" * 70)
    logger.info("Model Statistics")
    logger.info("=" * 70)
    
    stats = detector.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.2f}")
        else:
            logger.info(f"   {key}: {value}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Training and Testing Complete!")
    logger.info("=" * 70)
    logger.info("\n📝 Next Steps:")
    logger.info("   1. Start the API: python api_mindspore.py")
    logger.info("   2. Run tests: python test_mindspore.py")
    logger.info("   3. Check logs: phishing_detector_mindspore.log")
    logger.info("   4. Models saved in: data/mindspore_models/")
    logger.info("=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)