"""
MindSpore-based Phishing Ad & Referral Channel Detector
Uses deep neural networks for superior detection accuracy
"""

import math
import pandas as pd
import numpy as np
import requests
import whois
import json
import logging
import pickle
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# MindSpore imports
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.callback import TimeMonitor
from mindspore.nn.metrics import Accuracy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phishing_detector_mindspore.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set MindSpore context (choose based on your hardware)
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")  # For CPU
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # For GPU
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")  # For Ascend
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")  # Dynamic mode for debugging


class URLFeatureExtractor:
    """Extract features from URLs"""
    
    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy"""
        if not text:
            return 0.0
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        probabilities = [float(count) / len(text) for count in char_counts.values()]
        entropy = -sum([p * math.log2(p) for p in probabilities if p > 0])
        return entropy
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            domain = url.split('//')[-1].split('/')[0].split('?')[0].split(':')[0]
            return domain.lower()
        except:
            return ""
    
    @staticmethod
    def get_domain_age(domain: str) -> int:
        """Get domain age via WhoisXML API"""
        try:
            # Use WhoisXML API for reliable WHOIS data
            api_key = os.environ.get('WHOISXML_API_KEY')
            if not api_key:
                logger.debug("WHOISXML_API_KEY not set, cannot lookup domain age")
                return -1
            
            url = "https://www.whoisxmlapi.com/whoisserver/WhoisService"
            params = {
                'apiKey': api_key,
                'domainName': domain,
                'outputFormat': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            whois_record = data.get('WhoisRecord', {})
            
            # Use pre-calculated estimatedDomainAge if available
            age_days = whois_record.get('estimatedDomainAge')
            if age_days is not None:
                return max(0, age_days)
            
            # Fallback to parsing created date
            created_date_raw = whois_record.get('createdDate')
            if created_date_raw:
                # Handle timezone format
                date_str = created_date_raw.replace('Z', '+00:00')
                if '+' in date_str and date_str[-4:].isdigit():
                    date_str = date_str[:-2] + ':' + date_str[-2:]
                creation_date = datetime.fromisoformat(date_str)
                age_days = (datetime.now(creation_date.tzinfo) - creation_date).days
                return max(0, age_days)
            
            return -1
        except Exception as e:
            logger.debug(f"WHOIS lookup failed for {domain}: {e}")
            return -1
    
    @staticmethod
    def extract_url_features(url: str, domain_age: Optional[int] = None) -> Dict[str, float]:
        """Extract comprehensive URL features"""
        domain = URLFeatureExtractor.extract_domain(url)
        
        features = {
            'url_length': len(url),
            'url_entropy': URLFeatureExtractor.calculate_entropy(url),
            'domain_age': domain_age if domain_age is not None else URLFeatureExtractor.get_domain_age(domain),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'uses_https': 1.0 if url.startswith('https://') else 0.0,
            'has_ip': 1.0 if any(char.isdigit() for char in domain.split('.')[0] if domain) else 0.0,
            'suspicious_tld': 1.0 if any(tld in domain for tld in ['.xyz', '.tk', '.ml', '.ga', '.gq', '.cf']) else 0.0,
        }
        
        # Normalize numerical features
        if features['domain_age'] > 0:
            features['domain_age_normalized'] = min(1.0, features['domain_age'] / 3650.0)  # Normalize to 10 years
        else:
            features['domain_age_normalized'] = 0.0
        
        features['url_length_normalized'] = min(1.0, features['url_length'] / 200.0)
        features['url_entropy_normalized'] = min(1.0, features['url_entropy'] / 5.0)
        
        return features


class TextFeatureExtractor:
    """Extract features from ad text"""
    
    # Suspicious keywords
    URGENT_WORDS = ['urgent', 'verify', 'suspended', 'alert', 'warning', 'act now', 'immediate', 'expire']
    PHISHING_WORDS = ['click', 'claim', 'free', 'winner', 'prize', 'congratulations', 'won', 'bonus']
    FINANCIAL_WORDS = ['account', 'bank', 'payment', 'momo', 'money', 'transfer', 'pin', 'password']
    
    @staticmethod
    def extract_text_features(text: str) -> Dict[str, float]:
        """Extract text-based features"""
        text_lower = text.lower()
        
        features = {
            'text_length': len(text),
            'num_exclamation': text.count('!'),
            'num_question': text.count('?'),
            'num_uppercase': sum(1 for c in text if c.isupper()),
            'has_urgent_words': sum(1 for word in TextFeatureExtractor.URGENT_WORDS if word in text_lower),
            'has_phishing_words': sum(1 for word in TextFeatureExtractor.PHISHING_WORDS if word in text_lower),
            'has_financial_words': sum(1 for word in TextFeatureExtractor.FINANCIAL_WORDS if word in text_lower),
            'all_caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0.0,
        }
        
        # Normalize
        features['text_length_normalized'] = min(1.0, features['text_length'] / 500.0)
        features['num_exclamation_normalized'] = min(1.0, features['num_exclamation'] / 5.0)
        
        return features


class PhishingDetectorNN(nn.Cell):
    """
    Deep Neural Network for Phishing Detection using MindSpore
    Architecture: Multi-layer perceptron with dropout and batch normalization
    """
    
    def __init__(self, input_size: int = 30, hidden_sizes: List[int] = [128, 64, 32], dropout_rate: float = 0.3):
        super(PhishingDetectorNN, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Dense(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(keep_prob=1-dropout_rate))
            prev_size = hidden_size
        
        # Output layer (binary classification)
        layers.append(nn.Dense(prev_size, 2))
        
        self.network = nn.SequentialCell(layers)
        self.softmax = nn.Softmax(axis=1)
    
    def construct(self, x):
        """Forward pass"""
        logits = self.network(x)
        return logits


class AnomalyDetectorNN(nn.Cell):
    """
    Autoencoder for anomaly detection in transaction patterns
    """
    
    def __init__(self, input_size: int = 20, encoding_size: int = 10):
        super(AnomalyDetectorNN, self).__init__()
        
        # Encoder
        self.encoder = nn.SequentialCell([
            nn.Dense(input_size, 32),
            nn.ReLU(),
            nn.Dense(32, 16),
            nn.ReLU(),
            nn.Dense(16, encoding_size),
        ])
        
        # Decoder
        self.decoder = nn.SequentialCell([
            nn.Dense(encoding_size, 16),
            nn.ReLU(),
            nn.Dense(16, 32),
            nn.ReLU(),
            nn.Dense(32, input_size),
        ])
    
    def construct(self, x):
        """Forward pass - reconstruction"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MindSporePhishingDetector:
    """Main phishing detector using MindSpore"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the detector"""
        self.config = self._load_config(config_path)
        self.url_extractor = URLFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        
        # Neural networks
        self.classifier = None
        self.anomaly_detector = None
        self.is_trained = False
        
        # Feature configuration
        self.feature_names = []
        self.feature_mean = None
        self.feature_std = None
        
        # Watchlist
        self.watchlist = set()
        self._load_watchlist()
        
        # Detection history
        self.detection_history = []
        
        # Try to load existing models
        self._load_models()
        
        logger.info("MindSpore Phishing Detector initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            "model": {
                "hidden_sizes": [128, 64, 32],
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 32
            },
            "risk_thresholds": {
                "high_risk_probability": 0.8,
                "medium_risk_probability": 0.5,
                "domain_age_days": 90,
                "entropy_threshold": 3.5
            },
            "notifications": {
                "sms_enabled": True,
                "webhook_enabled": True,
                "webhook_url": "https://your-webhook-url.com/alert"
            },
            "storage": {
                "watchlist_file": "data/watchlist.json",
                "models_dir": "data/mindspore_models/",
                "checkpoint_prefix": "phishing_detector"
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
    
    def _load_watchlist(self):
        """Load watchlist from file"""
        watchlist_file = self.config["storage"]["watchlist_file"]
        if os.path.exists(watchlist_file):
            try:
                with open(watchlist_file, 'r') as f:
                    self.watchlist = set(json.load(f))
                logger.info(f"Loaded {len(self.watchlist)} domains from watchlist")
            except Exception as e:
                logger.error(f"Error loading watchlist: {e}")
    
    def _save_watchlist(self):
        """Save watchlist to file"""
        watchlist_file = self.config["storage"]["watchlist_file"]
        os.makedirs(os.path.dirname(watchlist_file), exist_ok=True)
        try:
            with open(watchlist_file, 'w') as f:
                json.dump(list(self.watchlist), f, indent=2)
            logger.info(f"Watchlist saved with {len(self.watchlist)} domains")
        except Exception as e:
            logger.error(f"Error saving watchlist: {e}")
    
    def _extract_all_features(self, signals: Dict) -> Dict[str, float]:
        """Extract all features from signals"""
        features = {}
        
        # URL features
        url_features = self.url_extractor.extract_url_features(
            signals.get('referrer_url', ''),
            signals.get('landing_domain_age')
        )
        features.update(url_features)
        
        # Text features
        text_features = self.text_extractor.extract_text_features(
            signals.get('ad_text', '')
        )
        features.update(text_features)
        
        # Behavioral features
        funnel_path = signals.get('funnel_path', [])
        features['funnel_depth'] = len(funnel_path)
        features['has_form_submission'] = 1.0 if any('form' in str(p).lower() for p in funnel_path) else 0.0
        features['new_recipient'] = 1.0 if signals.get('new_recipient', False) else 0.0
        features['spike_in_conversions'] = 1.0 if signals.get('spike_in_conversions', False) else 0.0
        features['same_ad_many_victims'] = 1.0 if signals.get('same_ad_used_by_many_victims', False) else 0.0
        
        return features
    
    def _prepare_features(self, features_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to numpy array"""
        if not self.feature_names:
            # First time - establish feature order
            self.feature_names = sorted(features_dict.keys())
        
        # Create feature vector in consistent order
        feature_vector = np.array([features_dict.get(name, 0.0) for name in self.feature_names], dtype=np.float32)
        
        # Normalize if we have statistics
        if self.feature_mean is not None and self.feature_std is not None:
            feature_vector = (feature_vector - self.feature_mean) / (self.feature_std + 1e-8)
        
        return feature_vector
    
    def train(self, training_data: List[Dict], save_models: bool = True):
        """
        Train the MindSpore models
        
        Args:
            training_data: List of dicts with 'signals' and 'is_phishing' keys
            save_models: Whether to save trained models
        """
        logger.info(f"Training with {len(training_data)} samples using MindSpore...")
        
        try:
            # Extract features
            all_features = []
            all_labels = []
            
            for item in training_data:
                features = self._extract_all_features(item['signals'])
                all_features.append(features)
                all_labels.append(item['is_phishing'])
            
            # Establish feature names from first sample
            self.feature_names = sorted(all_features[0].keys())
            
            # Convert to numpy arrays
            X = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in all_features], dtype=np.float32)
            y = np.array(all_labels, dtype=np.int32)
            
            # Calculate normalization statistics
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            
            # Normalize features
            X = (X - self.feature_mean) / (self.feature_std + 1e-8)
            
            # Convert to MindSpore tensors
            X_tensor = Tensor(X, ms.float32)
            y_tensor = Tensor(y, ms.int32)
            
            # Create dataset
            dataset_train = ms.dataset.NumpySlicesDataset(
                (X, y),
                column_names=["features", "label"],
                shuffle=True
            )
            dataset_train = dataset_train.batch(self.config['model']['batch_size'])
            
            # Initialize model
            input_size = len(self.feature_names)
            self.classifier = PhishingDetectorNN(
                input_size=input_size,
                hidden_sizes=self.config['model']['hidden_sizes'],
                dropout_rate=self.config['model']['dropout_rate']
            )
            
            # Loss and optimizer
            loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            optimizer = nn.Adam(self.classifier.trainable_params(), 
                              learning_rate=self.config['model']['learning_rate'])
            
            # Create model
            model = Model(self.classifier, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy': Accuracy()})
            
            # Callbacks
            models_dir = self.config['storage']['models_dir']
            os.makedirs(models_dir, exist_ok=True)
            
            # Configure checkpoint to save after each epoch
            steps_per_epoch = max(1, len(training_data) // self.config['model']['batch_size'])
            ckpt_config = CheckpointConfig(
                save_checkpoint_steps=steps_per_epoch,
                keep_checkpoint_max=5
            )
            ckpt_callback = ModelCheckpoint(
                prefix=self.config['storage']['checkpoint_prefix'],
                directory=models_dir,
                config=ckpt_config
            )
            
            # Train
            logger.info("Starting training...")
            logger.info(f"Saving checkpoints to: {models_dir}")
            model.train(
                epoch=self.config['model']['epochs'],
                train_dataset=dataset_train,
                callbacks=[ckpt_callback, LossMonitor(), TimeMonitor()]
            )
            
            # Verify checkpoint was saved
            checkpoint_files = [f for f in os.listdir(models_dir) if f.endswith('.ckpt')]
            if checkpoint_files:
                logger.info(f"✅ Checkpoint saved: {checkpoint_files[-1]}")
            else:
                logger.warning("⚠️  No checkpoint file created - attempting manual save...")
                # Manual checkpoint save as fallback
                manual_ckpt_path = os.path.join(models_dir, f"{self.config['storage']['checkpoint_prefix']}_final.ckpt")
                ms.save_checkpoint(self.classifier, manual_ckpt_path)
                logger.info(f"✅ Manual checkpoint saved: {manual_ckpt_path}")
            
            self.is_trained = True
            logger.info("Training complete!")
            
            # Save feature configuration
            if save_models:
                self._save_feature_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _save_feature_config(self):
        """Save feature configuration"""
        models_dir = self.config['storage']['models_dir']
        config_file = os.path.join(models_dir, 'feature_config.pkl')
        
        config = {
            'feature_names': self.feature_names,
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std
        }
        
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info("Feature configuration saved")
    
    def _load_models(self):
        """Load trained models from checkpoint"""
        models_dir = self.config['storage']['models_dir']
        config_file = os.path.join(models_dir, 'feature_config.pkl')
        
        if not os.path.exists(models_dir):
            logger.debug(f"Models directory not found: {models_dir}")
            return
        
        if os.path.exists(config_file):
            try:
                # Load feature configuration
                with open(config_file, 'rb') as f:
                    config = pickle.load(f)
                
                self.feature_names = config['feature_names']
                self.feature_mean = config['feature_mean']
                self.feature_std = config['feature_std']
                
                logger.info(f"Feature config loaded: {len(self.feature_names)} features")
                
                # Initialize model with correct input size
                input_size = len(self.feature_names)
                self.classifier = PhishingDetectorNN(
                    input_size=input_size,
                    hidden_sizes=self.config['model']['hidden_sizes'],
                    dropout_rate=self.config['model']['dropout_rate']
                )
                
                # Load checkpoint - find latest .ckpt file
                checkpoint_files = [f for f in os.listdir(models_dir) if f.endswith('.ckpt')]
                if checkpoint_files:
                    # Sort by filename to get latest
                    latest_ckpt = sorted(checkpoint_files)[-1]
                    ckpt_path = os.path.join(models_dir, latest_ckpt)
                    
                    logger.info(f"Loading checkpoint: {ckpt_path}")
                    param_dict = ms.load_checkpoint(ckpt_path)
                    ms.load_param_into_net(self.classifier, param_dict)
                    
                    self.is_trained = True
                    logger.info(f"✅ Model successfully loaded from {latest_ckpt}")
                else:
                    logger.warning(f"No checkpoint files found in {models_dir}")
                    logger.warning("Available files: " + ", ".join(os.listdir(models_dir)))
                    
            except Exception as e:
                logger.error(f"Could not load models: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.debug(f"Feature config not found: {config_file}")
    
    def detect_phishing(self, signals: Dict) -> Dict:
        """
        Detect phishing using MindSpore neural network
        
        Returns:
            Dict with risk_level, confidence, reasons, timestamp
        """
        detection_result = {
            'risk_level': 'LOW_RISK',
            'confidence': 0.0,
            'reasons': [],
            'timestamp': datetime.now().isoformat(),
            'signals': signals
        }
        
        try:
            logger.info(f"Analyzing signals for ad_id: {signals.get('ad_id', 'N/A')}")
            
            # Check watchlist first
            domain = self.url_extractor.extract_domain(signals.get('referrer_url', ''))
            if domain in self.watchlist:
                detection_result['risk_level'] = 'HIGH_RISK_WATCHLIST'
                detection_result['confidence'] = 1.0
                detection_result['reasons'].append(f"Domain '{domain}' is on watchlist")
                return detection_result
            
            # Extract and prepare features
            features_dict = self._extract_all_features(signals)
            features_array = self._prepare_features(features_dict)
            
            # Neural network prediction
            if self.is_trained and self.classifier is not None:
                # Convert to tensor and add batch dimension
                X = Tensor(features_array.reshape(1, -1), ms.float32)
                
                # Get prediction
                self.classifier.set_train(False)  # Set to eval mode
                logits = self.classifier(X)
                probs = ops.Softmax(axis=1)(logits)
                
                # Extract probabilities
                prob_phishing = float(probs[0, 1].asnumpy())
                prob_legit = float(probs[0, 0].asnumpy())
                
                detection_result['confidence'] = prob_phishing
                
                # Determine risk level based on probability
                high_threshold = self.config['risk_thresholds']['high_risk_probability']
                medium_threshold = self.config['risk_thresholds']['medium_risk_probability']
                
                if prob_phishing >= high_threshold:
                    detection_result['risk_level'] = 'HIGH_RISK_ML'
                    detection_result['reasons'].append(
                        f"Neural network flagged as phishing (confidence: {prob_phishing:.2%})"
                    )
                elif prob_phishing >= medium_threshold:
                    detection_result['risk_level'] = 'MEDIUM_RISK_ML'
                    detection_result['reasons'].append(
                        f"Neural network detected moderate risk (confidence: {prob_phishing:.2%})"
                    )
            
            # Rule-based checks (additional layer)
            if features_dict.get('domain_age', 0) < self.config['risk_thresholds']['domain_age_days']:
                detection_result['reasons'].append(
                    f"Very new domain (age: {features_dict['domain_age']} days)"
                )
                if detection_result['risk_level'] == 'LOW_RISK':
                    detection_result['risk_level'] = 'MEDIUM_RISK'
            
            if features_dict.get('has_urgent_words', 0) >= 2:
                detection_result['reasons'].append("Multiple urgent keywords detected in ad text")
            
            if features_dict.get('suspicious_tld', 0) == 1.0:
                detection_result['reasons'].append("Suspicious top-level domain (TLD)")
            
            # Transaction anomaly checks
            if signals.get('new_recipient') and signals.get('spike_in_conversions'):
                detection_result['reasons'].append("Transaction anomaly: new recipient + conversion spike")
                if 'HIGH_RISK' not in detection_result['risk_level']:
                    detection_result['risk_level'] = 'HIGH_RISK_ANOMALY'
            
            # Log detection
            self.detection_history.append(detection_result)
            logger.info(f"Detection complete: {detection_result['risk_level']} "
                       f"(confidence: {detection_result['confidence']:.2%})")
            
            return detection_result
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            detection_result['risk_level'] = 'ERROR'
            detection_result['reasons'].append(f"Detection error: {str(e)}")
            return detection_result
    
    def take_action(self, detection_result: Dict):
        """Execute actions based on detection results"""
        risk_level = detection_result['risk_level']
        signals = detection_result['signals']
        
        if risk_level.startswith('HIGH_RISK'):
            logger.warning(f"HIGH RISK DETECTED: {risk_level}")
            
            # SMS notification
            if self.config['notifications']['sms_enabled']:
                user_id = signals.get('user_id', 'unknown')
                self._send_sms_notification(user_id, detection_result)
            
            # Webhook notification
            if self.config['notifications']['webhook_enabled']:
                self._send_webhook_notification(detection_result)
            
            # Add to watchlist
            domain = self.url_extractor.extract_domain(signals.get('referrer_url', ''))
            if domain:
                self.add_to_watchlist(domain)
        
        elif risk_level.startswith('MEDIUM_RISK'):
            logger.info(f"MEDIUM RISK: Monitoring - {', '.join(detection_result['reasons'])}")
        else:
            logger.debug(f"LOW RISK: {signals.get('ad_id', 'N/A')}")
    
    def _send_sms_notification(self, user_id: str, detection_result: Dict):
        """Send SMS notification"""
        message = (
            "⚠️ Security Alert: We detected a suspicious ad that may attempt to steal your MoMo. "
            "Do not follow unknown links or provide personal information."
        )
        logger.info(f"SMS notification sent to user {user_id}")
        # TODO: Integrate with SMS API
    
    def _send_webhook_notification(self, detection_result: Dict):
        """Send webhook notification"""
        try:
            webhook_url = self.config['notifications']['webhook_url']
            payload = {
                'timestamp': detection_result['timestamp'],
                'risk_level': detection_result['risk_level'],
                'confidence': detection_result['confidence'],
                'reasons': detection_result['reasons']
            }
            logger.info(f"Webhook notification prepared: {payload}")
            # TODO: Uncomment for production
            # response = requests.post(webhook_url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    def add_to_watchlist(self, domain: str):
        """Add domain to watchlist"""
        if domain and domain not in self.watchlist:
            self.watchlist.add(domain)
            self._save_watchlist()
            logger.info(f"Added '{domain}' to watchlist")
    
    def remove_from_watchlist(self, domain: str):
        """Remove domain from watchlist"""
        if domain in self.watchlist:
            self.watchlist.remove(domain)
            self._save_watchlist()
            logger.info(f"Removed '{domain}' from watchlist")
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {'total': 0, 'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
        
        stats = {
            'total': len(self.detection_history),
            'high_risk': sum(1 for d in self.detection_history if 'HIGH_RISK' in d['risk_level']),
            'medium_risk': sum(1 for d in self.detection_history if 'MEDIUM_RISK' in d['risk_level']),
            'low_risk': sum(1 for d in self.detection_history if d['risk_level'] == 'LOW_RISK'),
            'watchlist_size': len(self.watchlist),
            'avg_confidence': np.mean([d['confidence'] for d in self.detection_history]) if self.detection_history else 0.0,
            'is_trained': self.is_trained
        }
        return stats