"""
Proactive Pre-Transaction Warning & User Prompting Service
Uses MindSpore for cohort selection, risk uplift modeling, and campaign detection
"""

import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

# MindSpore imports
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('proactive_warning_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set MindSpore context
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


@dataclass
class UserProfile:
    """User profile for vulnerability assessment"""
    user_id: str
    recent_clicks: int
    recent_calls: int
    new_device: bool
    device_age_days: int
    transaction_count_7d: int
    avg_transaction_amount: float
    account_age_days: int
    risky_interactions_7d: int
    last_warning_timestamp: Optional[datetime] = None
    warning_count_30d: int = 0
    opted_in_for_warnings: bool = True


@dataclass
class ScamCampaign:
    """Detected scam campaign"""
    campaign_id: str
    campaign_type: str  # 'call', 'sms', 'ad'
    start_timestamp: datetime
    affected_users: Set[str]
    suspicious_numbers: Set[str]
    volume: int
    risk_score: float
    description: str
    active: bool = True


class CohortSelectionModel(nn.Cell):
    """
    MindSpore model for vulnerable user cohort selection
    Identifies users at high risk based on behavioral patterns
    """

    def __init__(self, input_size: int = 15, hidden_sizes: List[int] = [64, 32, 16]):
        super(CohortSelectionModel, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            prev_size = hidden_size

        # Output: vulnerability score (0-1)
        layers.append(nn.Dense(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


class RiskUpliftModel(nn.Cell):
    """
    MindSpore model for risk uplift prediction
    Predicts increase in risk given active campaign
    """

    def __init__(self, input_size: int = 20, hidden_sizes: List[int] = [128, 64, 32]):
        super(RiskUpliftModel, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.3))
            prev_size = hidden_size

        # Output: risk uplift score (0-1)
        layers.append(nn.Dense(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


class CampaignDetectionModel(nn.Cell):
    """
    MindSpore model for anomaly detection in call/transaction patterns
    Detects active scam campaigns
    """

    def __init__(self, sequence_length: int = 24, input_features: int = 10):
        super(CampaignDetectionModel, self).__init__()

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Dense layers for classification
        self.fc = nn.SequentialCell([
            nn.Dense(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Dense(32, 1),
            nn.Sigmoid()
        ])

    def construct(self, x):
        # x shape: (batch, sequence_length, input_features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class SMSNotificationManager:
    """
    Manages SMS notifications with consent and frequency controls
    """

    def __init__(self, config: Dict):
        self.config = config
        self.notification_history = defaultdict(list)
        self.daily_limits = config.get('sms_limits', {
            'max_per_user_per_day': 3,
            'max_per_user_per_week': 10,
            'cooldown_hours': 6
        })
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load SMS templates"""
        return {
            'scam_warning': (
                "⚠️ FRAUD ALERT: We detected a scam campaign targeting customers. "
                "DO NOT share your PIN or OTP. If you receive suspicious calls, "
                "ignore them or call our helpline: 123. - {service_name}"
            ),
            'high_risk_transaction': (
                "⚠️ SECURITY: Unusual activity detected. For your next transaction, "
                "we'll require additional verification. Stay safe! - {service_name}"
            ),
            'campaign_alert': (
                "⚠️ ALERT: Active {campaign_type} scam detected. Be cautious of "
                "unexpected calls/messages. Never share PIN/OTP. Call 123 if unsure. - {service_name}"
            ),
            'targeted_warning': (
                "⚠️ WARNING: You may be targeted by scammers. Extra verification "
                "enabled temporarily. DO NOT share sensitive info. Help: 123. - {service_name}"
            )
        }

    def can_send_notification(self, user_id: str, profile: UserProfile) -> Tuple[bool, str]:
        """Check if notification can be sent to user"""

        # Check opt-in consent
        if not profile.opted_in_for_warnings:
            return False, "User opted out of warnings"

        # Check daily limit
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        recent_notifications = [
            ts for ts in self.notification_history[user_id]
            if ts > today_start
        ]

        if len(recent_notifications) >= self.daily_limits['max_per_user_per_day']:
            return False, "Daily limit reached"

        # Check cooldown
        if recent_notifications:
            last_notification = max(recent_notifications)
            cooldown_end = last_notification + timedelta(hours=self.daily_limits['cooldown_hours'])
            if now < cooldown_end:
                return False, f"Cooldown active until {cooldown_end}"

        # Check weekly limit
        week_start = now - timedelta(days=7)
        week_notifications = [
            ts for ts in self.notification_history[user_id]
            if ts > week_start
        ]

        if len(week_notifications) >= self.daily_limits['max_per_user_per_week']:
            return False, "Weekly limit reached"

        return True, "OK"

    def send_notification(
        self,
        user_id: str,
        phone_number: str,
        template_key: str,
        context: Dict = None
    ) -> Dict:
        """Send SMS notification"""

        if template_key not in self.templates:
            return {
                'success': False,
                'error': f'Unknown template: {template_key}'
            }

        # Get message
        message = self.templates[template_key].format(
            service_name=self.config.get('service_name', 'MoMo'),
            **(context or {})
        )

        # Record notification
        self.notification_history[user_id].append(datetime.now())

        # In production, integrate with SMS gateway (Twilio, Africa's Talking, etc.)
        logger.info(f"SMS sent to {user_id} ({phone_number}): {message}")

        return {
            'success': True,
            'user_id': user_id,
            'phone_number': phone_number,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }


class ProactiveWarningService:
    """
    Main Proactive Warning Service with MindSpore models
    """

    def __init__(self, config_path: str = "config/warning_config.json"):
        """Initialize the service"""
        self.config = self._load_config(config_path)

        # Initialize models
        self.cohort_model = None
        self.risk_uplift_model = None
        self.campaign_detection_model = None
        self.models_loaded = False

        # Initialize managers
        self.sms_manager = SMSNotificationManager(self.config)

        # Active campaigns tracking
        self.active_campaigns: Dict[str, ScamCampaign] = {}

        # User profiles cache
        self.user_profiles: Dict[str, UserProfile] = {}

        # Statistics
        self.stats = {
            'campaigns_detected': 0,
            'warnings_sent': 0,
            'users_protected': set(),
            'false_positives': 0,
            'true_positives': 0
        }

        # Load models if available
        self._load_models()

        logger.info("Proactive Warning Service initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            "service_name": "MoMo Fraud Prevention",
            "thresholds": {
                "cohort_vulnerability_score": 0.7,
                "risk_uplift_threshold": 0.6,
                "campaign_anomaly_score": 0.8,
                "min_campaign_volume": 50
            },
            "sms_limits": {
                "max_per_user_per_day": 3,
                "max_per_user_per_week": 10,
                "cooldown_hours": 6
            },
            "stricter_verification": {
                "enabled": True,
                "duration_hours": 24,
                "verification_level": "high"
            },
            "models": {
                "cohort_model_path": "data/models/cohort_model.ckpt",
                "risk_uplift_model_path": "data/models/risk_uplift_model.ckpt",
                "campaign_model_path": "data/models/campaign_model.ckpt"
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

    def _load_models(self):
        """Load trained MindSpore models"""
        try:
            model_config = self.config['models']

            # Load cohort selection model
            if os.path.exists(model_config['cohort_model_path']):
                self.cohort_model = CohortSelectionModel()
                # In production: ms.load_checkpoint(path, self.cohort_model)
                logger.info("Cohort selection model loaded")

            # Load risk uplift model
            if os.path.exists(model_config['risk_uplift_model_path']):
                self.risk_uplift_model = RiskUpliftModel()
                # In production: ms.load_checkpoint(path, self.risk_uplift_model)
                logger.info("Risk uplift model loaded")

            # Load campaign detection model
            if os.path.exists(model_config['campaign_model_path']):
                self.campaign_detection_model = CampaignDetectionModel()
                # In production: ms.load_checkpoint(path, self.campaign_detection_model)
                logger.info("Campaign detection model loaded")

            self.models_loaded = True

        except Exception as e:
            logger.warning(f"Models not loaded: {e}. Using heuristics.")
            self.models_loaded = False

    def detect_campaign(self, metrics: Dict) -> Optional[ScamCampaign]:
        """
        Detect active scam campaign using anomaly detection

        Args:
            metrics: Dictionary with call/transaction volumes, patterns

        Returns:
            ScamCampaign object if detected, None otherwise
        """

        # Extract metrics
        call_volume = metrics.get('call_volume_last_hour', 0)
        unique_targets = metrics.get('unique_targets', 0)
        suspicious_numbers = set(metrics.get('suspicious_numbers', []))
        pattern_anomaly_score = metrics.get('pattern_anomaly_score', 0.0)

        # Check thresholds
        min_volume = self.config['thresholds']['min_campaign_volume']
        anomaly_threshold = self.config['thresholds']['campaign_anomaly_score']

        # Rule-based detection (fallback if model not loaded)
        if call_volume > min_volume and pattern_anomaly_score > anomaly_threshold:
            campaign_id = f"CAMP_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            campaign = ScamCampaign(
                campaign_id=campaign_id,
                campaign_type=metrics.get('campaign_type', 'call'),
                start_timestamp=datetime.now(),
                affected_users=set(metrics.get('affected_users', [])),
                suspicious_numbers=suspicious_numbers,
                volume=call_volume,
                risk_score=pattern_anomaly_score,
                description=f"Detected {metrics.get('campaign_type', 'call')} campaign with {call_volume} incidents",
                active=True
            )

            self.active_campaigns[campaign_id] = campaign
            self.stats['campaigns_detected'] += 1

            logger.warning(f"Campaign detected: {campaign_id} - Volume: {call_volume}")

            return campaign

        return None

    def identify_vulnerable_cohort(self, users: List[Dict]) -> List[UserProfile]:
        """
        Identify vulnerable user cohort using ML model

        Args:
            users: List of user dictionaries with behavioral data

        Returns:
            List of UserProfile objects for vulnerable users
        """
        vulnerable_users = []
        threshold = self.config['thresholds']['cohort_vulnerability_score']

        for user_data in users:
            # Create user profile
            profile = UserProfile(
                user_id=user_data['user_id'],
                recent_clicks=user_data.get('recent_clicks', 0),
                recent_calls=user_data.get('recent_calls', 0),
                new_device=user_data.get('new_device', False),
                device_age_days=user_data.get('device_age_days', 365),
                transaction_count_7d=user_data.get('transaction_count_7d', 0),
                avg_transaction_amount=user_data.get('avg_transaction_amount', 0.0),
                account_age_days=user_data.get('account_age_days', 365),
                risky_interactions_7d=user_data.get('risky_interactions_7d', 0),
                warning_count_30d=user_data.get('warning_count_30d', 0),
                opted_in_for_warnings=user_data.get('opted_in_for_warnings', True)
            )

            # Calculate vulnerability score
            vulnerability_score = self._calculate_vulnerability_score(profile)

            if vulnerability_score >= threshold:
                vulnerable_users.append(profile)
                self.user_profiles[profile.user_id] = profile
                logger.info(f"Vulnerable user identified: {profile.user_id} (score: {vulnerability_score:.3f})")

        return vulnerable_users

    def _calculate_vulnerability_score(self, profile: UserProfile) -> float:
        """Calculate vulnerability score using heuristics or ML model"""

        # If model loaded, use it
        if self.models_loaded and self.cohort_model:
            features = self._extract_cohort_features(profile)
            features_tensor = Tensor(features, ms.float32).reshape(1, -1)
            score = self.cohort_model(features_tensor).asnumpy()[0][0]
            return float(score)

        # Fallback: heuristic scoring
        score = 0.0

        # Recent risky interactions (weight: 0.3)
        if profile.risky_interactions_7d > 0:
            score += min(0.3, profile.risky_interactions_7d * 0.1)

        # New device (weight: 0.2)
        if profile.new_device or profile.device_age_days < 7:
            score += 0.2

        # Recent clicks/calls (weight: 0.3)
        if profile.recent_clicks > 2 or profile.recent_calls > 3:
            score += 0.3

        # Account age (weight: 0.1)
        if profile.account_age_days < 30:
            score += 0.1

        # High transaction activity (weight: 0.1)
        if profile.transaction_count_7d > 10:
            score += 0.1

        return min(1.0, score)

    def _extract_cohort_features(self, profile: UserProfile) -> np.ndarray:
        """Extract features for cohort selection model"""
        features = np.array([
            profile.recent_clicks,
            profile.recent_calls,
            1.0 if profile.new_device else 0.0,
            profile.device_age_days / 365.0,  # Normalize
            profile.transaction_count_7d / 50.0,  # Normalize
            profile.avg_transaction_amount / 10000.0,  # Normalize
            profile.account_age_days / 365.0,  # Normalize
            profile.risky_interactions_7d,
            profile.warning_count_30d,
            # Additional features...
        ], dtype=np.float32)

        return features

    def send_proactive_warnings(
        self,
        campaign: ScamCampaign,
        vulnerable_users: List[UserProfile]
    ) -> Dict:
        """
        Send proactive warnings to vulnerable users

        Args:
            campaign: Detected scam campaign
            vulnerable_users: List of vulnerable user profiles

        Returns:
            Summary of warnings sent
        """
        results = {
            'warnings_sent': 0,
            'warnings_skipped': 0,
            'reasons': defaultdict(int),
            'notified_users': []
        }

        for profile in vulnerable_users:
            # Check if can send notification
            can_send, reason = self.sms_manager.can_send_notification(profile.user_id, profile)

            if not can_send:
                results['warnings_skipped'] += 1
                results['reasons'][reason] += 1
                logger.debug(f"Skipped warning for {profile.user_id}: {reason}")
                continue

            # Determine template based on campaign and user context
            template_key = self._select_template(campaign, profile)
            context = {
                'campaign_type': campaign.campaign_type
            }

            # Send SMS (in production, get actual phone number from user database)
            phone_number = f"+237{profile.user_id[-9:]}"  # Placeholder

            result = self.sms_manager.send_notification(
                user_id=profile.user_id,
                phone_number=phone_number,
                template_key=template_key,
                context=context
            )

            if result['success']:
                results['warnings_sent'] += 1
                results['notified_users'].append(profile.user_id)
                self.stats['warnings_sent'] += 1
                self.stats['users_protected'].add(profile.user_id)

                # Enable stricter verification if configured
                if self.config['stricter_verification']['enabled']:
                    self._enable_stricter_verification(profile.user_id)

        logger.info(f"Warnings sent: {results['warnings_sent']}, skipped: {results['warnings_skipped']}")

        return results

    def _select_template(self, campaign: ScamCampaign, profile: UserProfile) -> str:
        """Select appropriate SMS template"""
        if profile.risky_interactions_7d > 5:
            return 'targeted_warning'
        elif campaign.risk_score > 0.9:
            return 'campaign_alert'
        else:
            return 'scam_warning'

    def _enable_stricter_verification(self, user_id: str):
        """Enable temporary stricter verification for user"""
        duration_hours = self.config['stricter_verification']['duration_hours']
        expires_at = datetime.now() + timedelta(hours=duration_hours)

        # In production: Update user's verification level in database
        logger.info(f"Stricter verification enabled for {user_id} until {expires_at}")

    def process_campaign_alert(self, campaign_data: Dict) -> Dict:
        """
        Main processing function for campaign detection and warning

        Args:
            campaign_data: Dictionary with campaign metrics and user data

        Returns:
            Results dictionary with actions taken
        """
        results = {
            'campaign_detected': False,
            'vulnerable_users_identified': 0,
            'warnings_sent': 0,
            'actions': []
        }

        # Step 1: Detect campaign
        campaign = self.detect_campaign(campaign_data.get('metrics', {}))

        if not campaign:
            results['actions'].append('No campaign detected')
            return results

        results['campaign_detected'] = True
        results['campaign_id'] = campaign.campaign_id
        results['actions'].append(f"Campaign detected: {campaign.campaign_id}")

        # Step 2: Identify vulnerable cohort
        users_data = campaign_data.get('users', [])
        vulnerable_users = self.identify_vulnerable_cohort(users_data)

        results['vulnerable_users_identified'] = len(vulnerable_users)
        results['actions'].append(f"Identified {len(vulnerable_users)} vulnerable users")

        if not vulnerable_users:
            return results

        # Step 3: Send proactive warnings
        warning_results = self.send_proactive_warnings(campaign, vulnerable_users)

        results['warnings_sent'] = warning_results['warnings_sent']
        results['warnings_skipped'] = warning_results['warnings_skipped']
        results['skip_reasons'] = dict(warning_results['reasons'])
        results['actions'].append(f"Sent {warning_results['warnings_sent']} warnings")

        return results

    def get_statistics(self) -> Dict:
        """Get service statistics"""
        return {
            'campaigns_detected': self.stats['campaigns_detected'],
            'warnings_sent': self.stats['warnings_sent'],
            'unique_users_protected': len(self.stats['users_protected']),
            'active_campaigns': len(self.active_campaigns),
            'models_loaded': self.models_loaded,
            'sms_limits': self.config['sms_limits']
        }


# Example usage
if __name__ == "__main__":
    # Initialize service
    service = ProactiveWarningService()

    # Simulate campaign detection
    campaign_data = {
        'metrics': {
            'call_volume_last_hour': 150,
            'unique_targets': 80,
            'suspicious_numbers': ['+237699999999', '+237688888888'],
            'pattern_anomaly_score': 0.92,
            'campaign_type': 'call',
            'affected_users': ['user_001', 'user_002', 'user_003']
        },
        'users': [
            {
                'user_id': 'user_001',
                'recent_clicks': 3,
                'recent_calls': 5,
                'new_device': True,
                'device_age_days': 2,
                'transaction_count_7d': 15,
                'avg_transaction_amount': 5000.0,
                'account_age_days': 20,
                'risky_interactions_7d': 7,
                'opted_in_for_warnings': True
            },
            {
                'user_id': 'user_002',
                'recent_clicks': 1,
                'recent_calls': 2,
                'new_device': False,
                'device_age_days': 180,
                'transaction_count_7d': 5,
                'avg_transaction_amount': 2000.0,
                'account_age_days': 365,
                'risky_interactions_7d': 1,
                'opted_in_for_warnings': True
            }
        ]
    }

    # Process campaign alert
    results = service.process_campaign_alert(campaign_data)

    print("\n" + "="*70)
    print("PROACTIVE WARNING SERVICE - TEST RESULTS")
    print("="*70)
    print(json.dumps(results, indent=2, default=str))
    print("="*70)

    # Get statistics
    stats = service.get_statistics()
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
