"""
Data Anonymization Engine

Advanced data anonymization and pseudonymization for fraud detection:
- Field-level anonymization (names, emails, phone numbers)
- Tokenization and hashing for reversible anonymization
- Differential privacy for statistical analysis
- Data masking for development/testing
- GDPR-compliant data handling
"""

import hashlib
import re
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class AnonymizationConfig:
    """Configuration for data anonymization"""
    preserve_format: bool = True
    reversible: bool = False
    salt: Optional[str] = None
    hash_algorithm: str = 'sha256'
    mask_char: str = '*'
    epsilon: float = 1.0  # For differential privacy

class DataAnonymizer:
    """
    Advanced data anonymization engine for fraud detection systems
    """
    
    def __init__(self, config: Optional[AnonymizationConfig] = None):
        self.config = config or AnonymizationConfig()
        self._token_map: Dict[str, str] = {}
        self._reverse_map: Dict[str, str] = {}
        
        # Patterns for PII detection
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b')
        self.ssn_pattern = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
        self.credit_card_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')
        self.ip_pattern = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        
    def anonymize_field(self, value: Any, field_type: str = 'text') -> str:
        """
        Anonymize a single field based on its type
        """
        if value is None:
            return None
            
        value_str = str(value)
        
        if field_type == 'email':
            return self._anonymize_email(value_str)
        elif field_type == 'phone':
            return self._anonymize_phone(value_str)
        elif field_type == 'name':
            return self._anonymize_name(value_str)
        elif field_type == 'ssn':
            return self._anonymize_ssn(value_str)
        elif field_type == 'credit_card':
            return self._anonymize_credit_card(value_str)
        elif field_type == 'ip_address':
            return self._anonymize_ip(value_str)
        elif field_type == 'address':
            return self._anonymize_address(value_str)
        else:
            return self._anonymize_text(value_str)
    
    def anonymize_record(self, record: Dict[str, Any], 
                        field_mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Anonymize an entire record with field-specific handling
        """
        field_mapping = field_mapping or {}
        anonymized = {}
        
        for field, value in record.items():
            field_type = field_mapping.get(field, 'text')
            anonymized[field] = self.anonymize_field(value, field_type)
            
        return anonymized
    
    def _anonymize_email(self, email: str) -> str:
        """Anonymize email address while preserving format"""
        if not self.email_pattern.match(email):
            return self._anonymize_text(email)
            
        local, domain = email.split('@', 1)
        
        if self.config.reversible:
            # Tokenize for reversible anonymization
            token = self._generate_token(email)
            return f"{token}@{domain}"
        else:
            # Mask local part
            if len(local) <= 2:
                masked_local = '*' * len(local)
            else:
                masked_local = local[0] + '*' * (len(local) - 2) + local[-1]
            return f"{masked_local}@{domain}"
    
    def _anonymize_phone(self, phone: str) -> str:
        """Anonymize phone number while preserving format"""
        match = self.phone_pattern.search(phone)
        if not match:
            return self._anonymize_text(phone)
            
        if self.config.reversible:
            return self._generate_token(phone)
        else:
            # Keep last 4 digits, mask the rest
            return re.sub(r'\d(?=\d{4})', self.config.mask_char, phone)
    
    def _anonymize_name(self, name: str) -> str:
        """Anonymize person name"""
        if self.config.reversible:
            return self._generate_token(name)
        else:
            # Keep first letter, mask the rest
            parts = name.split()
            anonymized_parts = []
            for part in parts:
                if len(part) <= 1:
                    anonymized_parts.append(self.config.mask_char)
                else:
                    anonymized_parts.append(part[0] + self.config.mask_char * (len(part) - 1))
            return ' '.join(anonymized_parts)
    
    def _anonymize_ssn(self, ssn: str) -> str:
        """Anonymize Social Security Number"""
        if self.ssn_pattern.match(ssn):
            return 'XXX-XX-' + ssn[-4:]
        return self._anonymize_text(ssn)
    
    def _anonymize_credit_card(self, card: str) -> str:
        """Anonymize credit card number"""
        # Remove non-digits
        digits = re.sub(r'\D', '', card)
        if len(digits) >= 13 and len(digits) <= 16:
            return self.config.mask_char * (len(digits) - 4) + digits[-4:]
        return self._anonymize_text(card)
    
    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address"""
        match = self.ip_pattern.match(ip)
        if match:
            parts = ip.split('.')
            return f"{parts[0]}.{parts[1]}.{self.config.mask_char}.{self.config.mask_char}"
        return self._anonymize_text(ip)
    
    def _anonymize_address(self, address: str) -> str:
        """Anonymize street address"""
        if self.config.reversible:
            return self._generate_token(address)
        else:
            # Replace numbers with X, keep street names
            words = address.split()
            anonymized_words = []
            for word in words:
                if word.isdigit():
                    anonymized_words.append('X' * len(word))
                else:
                    anonymized_words.append(word)
            return ' '.join(anonymized_words)
    
    def _anonymize_text(self, text: str) -> str:
        """General text anonymization"""
        if self.config.reversible:
            return self._generate_token(text)
        else:
            # Hash the text
            salt = self.config.salt or secrets.token_hex(16)
            hash_obj = hashlib.new(self.config.hash_algorithm)
            hash_obj.update(f"{text}{salt}".encode())
            return hash_obj.hexdigest()[:16]
    
    def _generate_token(self, value: str) -> str:
        """Generate reversible token for value"""
        if value in self._token_map:
            return self._token_map[value]
            
        # Generate unique token
        token = f"TOK_{secrets.token_hex(8)}"
        self._token_map[value] = token
        self._reverse_map[token] = value
        
        return token
    
    def deanonymize(self, token: str) -> Optional[str]:
        """Reverse anonymization for tokens"""
        return self._reverse_map.get(token)
    
    def add_differential_privacy_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """
        Add Laplace noise for differential privacy
        """
        import random
        scale = sensitivity / self.config.epsilon
        noise = random.laplace(0, scale)
        return value + noise
    
    def create_synthetic_dataset(self, original_data: List[Dict], 
                               sample_size: Optional[int] = None) -> List[Dict]:
        """
        Create synthetic dataset with same statistical properties
        """
        import numpy as np
        
        if sample_size is None:
            sample_size = len(original_data)
            
        synthetic_data = []
        
        # Analyze statistical properties
        field_stats = {}
        if original_data:
            for field in original_data[0].keys():
                values = [record.get(field) for record in original_data if record.get(field) is not None]
                if values and isinstance(values[0], (int, float)):
                    field_stats[field] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
        
        # Generate synthetic records
        for _ in range(sample_size):
            synthetic_record = {}
            for field, stats in field_stats.items():
                # Generate value with added noise for differential privacy
                noisy_mean = self.add_differential_privacy_noise(stats['mean'], stats['std'])
                noisy_std = max(0.1, self.add_differential_privacy_noise(stats['std'], stats['std']/10))
                
                value = np.random.normal(noisy_mean, noisy_std)
                value = max(stats['min'], min(stats['max'], value))  # Clamp to original range
                
                synthetic_record[field] = value
            
            synthetic_data.append(synthetic_record)
        
        return synthetic_data
    
    def get_anonymization_summary(self, original_data: Dict, 
                                 anonymized_data: Dict) -> Dict[str, Any]:
        """
        Generate summary of anonymization performed
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'fields_processed': len(original_data),
            'anonymization_method': 'reversible' if self.config.reversible else 'irreversible',
            'field_details': {}
        }
        
        for field in original_data:
            original_value = str(original_data[field])
            anonymized_value = str(anonymized_data[field])
            
            field_info = {
                'original_length': len(original_value),
                'anonymized_length': len(anonymized_value),
                'changed': original_value != anonymized_value
            }
            
            # Detect PII types
            if self.email_pattern.search(original_value):
                field_info['pii_type'] = 'email'
            elif self.phone_pattern.search(original_value):
                field_info['pii_type'] = 'phone'
            elif self.ssn_pattern.search(original_value):
                field_info['pii_type'] = 'ssn'
            elif self.credit_card_pattern.search(original_value):
                field_info['pii_type'] = 'credit_card'
            elif self.ip_pattern.search(original_value):
                field_info['pii_type'] = 'ip_address'
            
            summary['field_details'][field] = field_info
        
        return summary
    
    def save_token_mapping(self, filepath: str) -> None:
        """Save token mapping for reversible anonymization"""
        with open(filepath, 'w') as f:
            json.dump({
                'token_map': self._token_map,
                'reverse_map': self._reverse_map,
                'config': {
                    'preserve_format': self.config.preserve_format,
                    'reversible': self.config.reversible,
                    'salt': self.config.salt,
                    'hash_algorithm': self.config.hash_algorithm,
                    'mask_char': self.config.mask_char,
                    'epsilon': self.config.epsilon
                }
            }, f, indent=2)
    
    def load_token_mapping(self, filepath: str) -> None:
        """Load token mapping for reversible anonymization"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self._token_map = data['token_map']
            self._reverse_map = data['reverse_map']
            
            # Update config
            config_data = data['config']
            self.config.preserve_format = config_data['preserve_format']
            self.config.reversible = config_data['reversible']
            self.config.salt = config_data['salt']
            self.config.hash_algorithm = config_data['hash_algorithm']
            self.config.mask_char = config_data['mask_char']
            self.config.epsilon = config_data['epsilon']