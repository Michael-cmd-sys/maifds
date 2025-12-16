"""
Data Classification System

Automated data classification for privacy compliance:
- PII detection and classification
- Sensitivity level assignment
- Data category mapping
- ML-based classification
- Regulatory compliance tagging
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import json
import logging

logger = logging.getLogger(__name__)

class DataCategory(Enum):
    """Data categories for classification"""
    PERSONAL_IDENTIFIERS = "personal_identifiers"
    CONTACT_INFORMATION = "contact_information"
    FINANCIAL_DATA = "financial_data"
    TRANSACTION_DATA = "transaction_data"
    DEVICE_INFORMATION = "device_information"
    BEHAVIORAL_DATA = "behavioral_data"
    LOCATION_DATA = "location_data"
    BIOMETRIC_DATA = "biometric_data"
    HEALTH_DATA = "health_data"
    EMPLOYMENT_DATA = "employment_data"
    EDUCATION_DATA = "education_data"
    PUBLIC_RECORDS = "public_records"
    DERIVED_DATA = "derived_data"
    AGGREGATED_DATA = "aggregated_data"

class SensitivityLevel(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CRITICAL = "critical"

class RegulatoryScope(Enum):
    """Regulatory frameworks applicable to data"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"
    NONE = "none"

@dataclass
class ClassificationRule:
    """Rule for data classification"""
    rule_id: str
    name: str
    description: str
    patterns: List[str]  # Regex patterns
    category: DataCategory
    sensitivity: SensitivityLevel
    regulatory_scope: List[RegulatoryScope]
    confidence_threshold: float = 0.8
    active: bool = True

@dataclass
class DataClassification:
    """Classification result for data"""
    data_id: str
    category: DataCategory
    sensitivity: SensitivityLevel
    regulatory_scope: List[RegulatoryScope]
    confidence: float
    classification_date: datetime
    rules_matched: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataClassifier:
    """
    Automated data classification system
    """
    
    def __init__(self):
        self._rules: Dict[str, ClassificationRule] = {}
        self._classifications: Dict[str, DataClassification] = {}
        self._pii_patterns = self._initialize_pii_patterns()
        self._financial_patterns = self._initialize_financial_patterns()
        self._device_patterns = self._initialize_device_patterns()
        
        # Initialize default classification rules
        self._initialize_default_rules()
    
    def _initialize_pii_patterns(self) -> Dict[str, List[str]]:
        """Initialize PII detection patterns"""
        return {
            'email': [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            'phone': [
                r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                r'\b\+?[1-9]\d{1,14}\b'
            ],
            'ssn': [r'\b\d{3}-\d{2}-\d{4}\b'],
            'passport': [r'\b[A-Z]{2}\d{7}\b'],
            'driver_license': [r'\b[A-Z]{1,2}\d{6,8}\b'],
            'national_id': [r'\b\d{8,12}\b'],
            'ip_address': [r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'],
            'mac_address': [r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b']
        }
    
    def _initialize_financial_patterns(self) -> Dict[str, List[str]]:
        """Initialize financial data patterns"""
        return {
            'credit_card': [
                r'\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Visa
                r'\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # MasterCard
                r'\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b',  # American Express
                r'\b6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'  # Discover
            ],
            'bank_account': [r'\b\d{8,17}\b'],
            'routing_number': [r'\b\d{9}\b'],
            'iban': [r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}\b'],
            'swift': [r'\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b'],
            'crypto_wallet': [
                r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin
                r'\b0x[a-fA-F0-9]{40}\b'  # Ethereum
            ]
        }
    
    def _initialize_device_patterns(self) -> Dict[str, List[str]]:
        """Initialize device information patterns"""
        return {
            'user_agent': [r'Mozilla|Chrome|Safari|Firefox|Edge|Opera'],
            'device_id': [r'\b[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}\b'],
            'imei': [r'\b\d{15}\b'],
            'imsi': [r'\b\d{15}\b'],
            'fingerprint': [r'\b[a-fA-F0-9]{32,64}\b']
        }
    
    def _initialize_default_rules(self) -> None:
        """Initialize default classification rules"""
        
        # Email classification
        self._rules['email_pii'] = ClassificationRule(
            rule_id='email_pii',
            name='Email Address Classification',
            description='Classify email addresses as PII',
            patterns=self._pii_patterns['email'],
            category=DataCategory.CONTACT_INFORMATION,
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            regulatory_scope=[RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.9
        )
        
        # Phone classification
        self._rules['phone_pii'] = ClassificationRule(
            rule_id='phone_pii',
            name='Phone Number Classification',
            description='Classify phone numbers as PII',
            patterns=self._pii_patterns['phone'],
            category=DataCategory.CONTACT_INFORMATION,
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            regulatory_scope=[RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.8
        )
        
        # SSN classification
        self._rules['ssn_pii'] = ClassificationRule(
            rule_id='ssn_pii',
            name='Social Security Number Classification',
            description='Classify SSNs as highly sensitive PII',
            patterns=self._pii_patterns['ssn'],
            category=DataCategory.PERSONAL_IDENTIFIERS,
            sensitivity=SensitivityLevel.CRITICAL,
            regulatory_scope=[RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.95
        )
        
        # Credit card classification
        self._rules['credit_card_financial'] = ClassificationRule(
            rule_id='credit_card_financial',
            name='Credit Card Classification',
            description='Classify credit card numbers as financial data',
            patterns=self._financial_patterns['credit_card'],
            category=DataCategory.FINANCIAL_DATA,
            sensitivity=SensitivityLevel.CRITICAL,
            regulatory_scope=[RegulatoryScope.PCI_DSS, RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.9
        )
        
        # IP address classification
        self._rules['ip_device'] = ClassificationRule(
            rule_id='ip_device',
            name='IP Address Classification',
            description='Classify IP addresses as device information',
            patterns=self._pii_patterns['ip_address'],
            category=DataCategory.DEVICE_INFORMATION,
            sensitivity=SensitivityLevel.INTERNAL,
            regulatory_scope=[RegulatoryScope.GDPR],
            confidence_threshold=0.8
        )
        
        # Transaction amount classification
        self._rules['transaction_amount'] = ClassificationRule(
            rule_id='transaction_amount',
            name='Transaction Amount Classification',
            description='Classify monetary amounts as transaction data',
            patterns=[r'\$\d+(?:\.\d{2})?', r'\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)'],
            category=DataCategory.TRANSACTION_DATA,
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            regulatory_scope=[RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.7
        )
        
        # Location data classification
        self._rules['location_data'] = ClassificationRule(
            rule_id='location_data',
            name='Location Data Classification',
            description='Classify geographic coordinates and addresses',
            patterns=[
                r'\d+\s+[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Z]{2}\s*\d{5}',
                r'\-?\d+\.?\d*\,\s*\-?\d+\.?\d*'  # GPS coordinates
            ],
            category=DataCategory.LOCATION_DATA,
            sensitivity=SensitivityLevel.CONFIDENTIAL,
            regulatory_scope=[RegulatoryScope.GDPR, RegulatoryScope.CCPA],
            confidence_threshold=0.8
        )
    
    def classify_data(self, data: Any, data_id: Optional[str] = None) -> DataClassification:
        """
        Classify data based on patterns and rules
        """
        if data_id is None:
            data_id = f"data_{datetime.now().timestamp()}"
        
        data_str = str(data)
        matched_rules = []
        confidences = []
        categories = []
        sensitivities = []
        regulatory_scopes = set()
        
        # Check against all active rules
        for rule in self._rules.values():
            if not rule.active:
                continue
            
            rule_confidence = self._calculate_rule_confidence(data_str, rule)
            if rule_confidence >= rule.confidence_threshold:
                matched_rules.append(rule.rule_id)
                confidences.append(rule_confidence)
                categories.append(rule.category)
                sensitivities.append(rule.sensitivity)
                regulatory_scopes.update(rule.regulatory_scope)
        
        # Determine final classification
        if matched_rules:
            # Use highest confidence rule as primary
            max_confidence_idx = confidences.index(max(confidences))
            primary_category = categories[max_confidence_idx]
            primary_sensitivity = sensitivities[max_confidence_idx]
            overall_confidence = max(confidences)
        else:
            # Default classification for unclassified data
            primary_category = DataCategory.PUBLIC_RECORDS
            primary_sensitivity = SensitivityLevel.PUBLIC
            overall_confidence = 0.0
        
        classification = DataClassification(
            data_id=data_id,
            category=primary_category,
            sensitivity=primary_sensitivity,
            regulatory_scope=list(regulatory_scopes),
            confidence=overall_confidence,
            classification_date=datetime.now(),
            rules_matched=matched_rules,
            metadata={
                'data_length': len(data_str),
                'data_type': type(data).__name__,
                'matched_categories': [cat.value for cat in categories],
                'all_sensitivities': [sens.value for sens in sensitivities]
            }
        )
        
        self._classifications[data_id] = classification
        
        return classification
    
    def classify_record(self, record: Dict[str, Any]) -> Dict[str, DataClassification]:
        """
        Classify all fields in a record
        """
        classifications = {}
        
        for field_name, field_value in record.items():
            field_id = f"{record.get('id', 'record')}.{field_name}"
            classifications[field_name] = self.classify_data(field_value, field_id)
        
        return classifications
    
    def batch_classify(self, data_list: List[Any]) -> List[DataClassification]:
        """
        Classify multiple data items
        """
        return [self.classify_data(data, f"batch_{i}") for i, data in enumerate(data_list)]
    
    def get_classification(self, data_id: str) -> Optional[DataClassification]:
        """Get existing classification"""
        return self._classifications.get(data_id)
    
    def update_classification(self, data_id: str, classification: DataClassification) -> bool:
        """Update existing classification"""
        if data_id not in self._classifications:
            return False
        
        self._classifications[data_id] = classification
        return True
    
    def add_rule(self, rule: ClassificationRule) -> None:
        """Add a new classification rule"""
        self._rules[rule.rule_id] = rule
    
    def update_rule(self, rule_id: str, **kwargs) -> bool:
        """Update an existing rule"""
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        return True
    
    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a classification rule"""
        if rule_id not in self._rules:
            return False
        
        self._rules[rule_id].active = False
        return True
    
    def _calculate_rule_confidence(self, data: str, rule: ClassificationRule) -> float:
        """Calculate confidence score for a rule match"""
        if not rule.patterns:
            return 0.0
        
        max_confidence = 0.0
        
        for pattern in rule.patterns:
            try:
                matches = re.findall(pattern, data, re.IGNORECASE)
                if matches:
                    # Base confidence on number and quality of matches
                    pattern_confidence = min(1.0, len(matches) * 0.3)
                    
                    # Adjust confidence based on pattern specificity
                    if len(pattern) > 20:  # More specific patterns
                        pattern_confidence *= 1.2
                    
                    max_confidence = max(max_confidence, pattern_confidence)
            except re.error:
                logger.warning(f"Invalid regex pattern in rule {rule.rule_id}: {pattern}")
                continue
        
        return min(1.0, max_confidence)
    
    def detect_pii_types(self, data: str) -> List[str]:
        """Detect specific PII types in data"""
        pii_types = []
        
        for pii_type, patterns in self._pii_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, data, re.IGNORECASE):
                        pii_types.append(pii_type)
                        break
                except re.error:
                    continue
        
        return pii_types
    
    def get_data_summary(self, data_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of data classification"""
        classification = self.get_classification(data_id)
        if not classification:
            return {}
        
        return {
            'data_id': data_id,
            'category': classification.category.value,
            'sensitivity': classification.sensitivity.value,
            'regulatory_scope': [scope.value for scope in classification.regulatory_scope],
            'confidence': classification.confidence,
            'classification_date': classification.classification_date.isoformat(),
            'rules_matched': classification.rules_matched,
            'metadata': classification.metadata,
            'requires_consent': classification.sensitivity in [SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED, SensitivityLevel.CRITICAL],
            'requires_encryption': classification.sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CRITICAL],
            'audit_required': classification.sensitivity in [SensitivityLevel.RESTRICTED, SensitivityLevel.CRITICAL]
        }
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics"""
        if not self._classifications:
            return {}
        
        # Category distribution
        category_counts = {}
        sensitivity_counts = {}
        regulatory_counts = {}
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
        
        for classification in self._classifications.values():
            # Category counts
            cat = classification.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Sensitivity counts
            sens = classification.sensitivity.value
            sensitivity_counts[sens] = sensitivity_counts.get(sens, 0) + 1
            
            # Regulatory scope counts
            for scope in classification.regulatory_scope:
                scope_key = scope.value
                regulatory_counts[scope_key] = regulatory_counts.get(scope_key, 0) + 1
            
            # Confidence distribution
            if classification.confidence >= 0.8:
                confidence_distribution['high'] += 1
            elif classification.confidence >= 0.5:
                confidence_distribution['medium'] += 1
            elif classification.confidence > 0:
                confidence_distribution['low'] += 1
            else:
                confidence_distribution['none'] += 1
        
        return {
            'total_classifications': len(self._classifications),
            'active_rules': len([r for r in self._rules.values() if r.active]),
            'category_distribution': category_counts,
            'sensitivity_distribution': sensitivity_counts,
            'regulatory_distribution': regulatory_counts,
            'confidence_distribution': confidence_distribution,
            'average_confidence': sum(c.confidence for c in self._classifications.values()) / len(self._classifications)
        }
    
    def export_classifications(self, format: str = 'json') -> str:
        """Export all classifications"""
        data = {
            'export_date': datetime.now().isoformat(),
            'total_classifications': len(self._classifications),
            'classifications': [
                {
                    'data_id': cls.data_id,
                    'category': cls.category.value,
                    'sensitivity': cls.sensitivity.value,
                    'regulatory_scope': [scope.value for scope in cls.regulatory_scope],
                    'confidence': cls.confidence,
                    'classification_date': cls.classification_date.isoformat(),
                    'rules_matched': cls.rules_matched,
                    'metadata': cls.metadata
                }
                for cls in self._classifications.values()
            ]
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_rule(self, rule_id: str) -> Optional[ClassificationRule]:
        """Get classification rule by ID"""
        return self._rules.get(rule_id)
    
    def list_rules(self) -> List[ClassificationRule]:
        """List all classification rules"""
        return list(self._rules.values())
    
    def search_classifications(self, category: Optional[DataCategory] = None,
                            sensitivity: Optional[SensitivityLevel] = None,
                            regulatory_scope: Optional[RegulatoryScope] = None,
                            min_confidence: Optional[float] = None) -> List[DataClassification]:
        """Search classifications with filters"""
        results = []
        
        for classification in self._classifications.values():
            if category and classification.category != category:
                continue
            if sensitivity and classification.sensitivity != sensitivity:
                continue
            if regulatory_scope and regulatory_scope not in classification.regulatory_scope:
                continue
            if min_confidence and classification.confidence < min_confidence:
                continue
            
            results.append(classification)
        
        return results