"""
Consent Management System

Comprehensive consent tracking and management for GDPR compliance:
- Granular consent tracking (data processing, analytics, sharing)
- Consent withdrawal and history
- Purpose-based consent management
- Automated consent expiration
- Audit trail for consent changes
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

class ConsentPurpose(Enum):
    """GDPR-defined purposes for data processing"""
    PERFORMANCE_OF_CONTRACT = "performance_of_contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"
    CONSENT = "consent"

class ConsentStatus(Enum):
    """Consent status values"""
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"

@dataclass
class ConsentRecord:
    """Individual consent record"""
    consent_id: str
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    granted_at: datetime
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    legal_basis: str = ""
    data_categories: List[str] = field(default_factory=list)
    processing_activities: List[str] = field(default_factory=list)
    third_parties: List[str] = field(default_factory=list)

@dataclass
class ConsentTemplate:
    """Template for consent requests"""
    template_id: str
    name: str
    description: str
    purpose: ConsentPurpose
    legal_basis: str
    data_categories: List[str]
    processing_activities: List[str]
    third_parties: List[str]
    retention_period_days: int
    required: bool = False
    version: str = "1.0"

class ConsentManager:
    """
    Comprehensive consent management system
    """
    
    def __init__(self):
        self._consents: Dict[str, ConsentRecord] = {}
        self._templates: Dict[str, ConsentTemplate] = {}
        self._user_consents: Dict[str, Set[str]] = {}  # user_id -> consent_ids
        self._default_templates = self._create_default_templates()
        
    def _create_default_templates(self) -> Dict[str, ConsentTemplate]:
        """Create default consent templates for fraud detection"""
        templates = {}
        
        # Fraud detection consent
        templates['fraud_detection'] = ConsentTemplate(
            template_id='fraud_detection',
            name='Fraud Detection and Prevention',
            description='Processing your data to detect and prevent fraudulent activities',
            purpose=ConsentPurpose.LEGITIMATE_INTERESTS,
            legal_basis='Legitimate interest in fraud prevention',
            data_categories=['transaction_data', 'device_info', 'behavioral_data'],
            processing_activities=['fraud_analysis', 'risk_scoring', 'pattern_detection'],
            third_parties=['payment_processors', 'credit_bureaus'],
            retention_period_days=2555,  # 7 years
            required=True
        )
        
        # Analytics consent
        templates['analytics'] = ConsentTemplate(
            template_id='analytics',
            name='Service Analytics',
            description='Anonymous usage data to improve our services',
            purpose=ConsentPurpose.LEGITIMATE_INTERESTS,
            legal_basis='Legitimate interest in service improvement',
            data_categories=['usage_data', 'performance_data'],
            processing_activities=['statistical_analysis', 'service_optimization'],
            third_parties=[],
            retention_period_days=365,
            required=False
        )
        
        # Marketing consent
        templates['marketing'] = ConsentTemplate(
            template_id='marketing',
            name='Marketing Communications',
            description='Receive information about our products and services',
            purpose=ConsentPurpose.CONSENT,
            legal_basis='Explicit consent',
            data_categories=['contact_data', 'preferences'],
            processing_activities=['email_marketing', 'personalization'],
            third_parties=['marketing_platforms'],
            retention_period_days=730,
            required=False
        )
        
        # Data sharing consent
        templates['data_sharing'] = ConsentTemplate(
            template_id='data_sharing',
            name='Data Sharing with Partners',
            description='Share relevant data with trusted partners for enhanced security',
            purpose=ConsentPurpose.LEGITIMATE_INTERESTS,
            legal_basis='Legitimate interest in collaborative security',
            data_categories=['risk_indicators', 'fraud_patterns'],
            processing_activities=['secure_data_sharing', 'threat_intelligence'],
            third_parties=['security_partners', 'law_enforcement'],
            retention_period_days=1825,  # 5 years
            required=False
        )
        
        return templates
    
    def create_consent_request(self, user_id: str, template_id: str, 
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new consent request
        """
        if template_id not in self._templates and template_id not in self._default_templates:
            raise ValueError(f"Unknown consent template: {template_id}")
        
        template = self._templates.get(template_id) or self._default_templates[template_id]
        
        consent_id = str(uuid.uuid4())
        consent = ConsentRecord(
            consent_id=consent_id,
            user_id=user_id,
            purpose=template.purpose,
            status=ConsentStatus.PENDING,
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=template.retention_period_days),
            metadata=metadata or {},
            legal_basis=template.legal_basis,
            data_categories=template.data_categories,
            processing_activities=template.processing_activities,
            third_parties=template.third_parties
        )
        
        self._consents[consent_id] = consent
        
        if user_id not in self._user_consents:
            self._user_consents[user_id] = set()
        self._user_consents[user_id].add(consent_id)
        
        return consent_id
    
    def grant_consent(self, consent_id: str, user_id: str) -> bool:
        """
        Grant consent for a specific request
        """
        if consent_id not in self._consents:
            return False
        
        consent = self._consents[consent_id]
        if consent.user_id != user_id:
            return False
        
        consent.status = ConsentStatus.GRANTED
        consent.granted_at = datetime.now()
        
        return True
    
    def deny_consent(self, consent_id: str, user_id: str) -> bool:
        """
        Deny consent for a specific request
        """
        if consent_id not in self._consents:
            return False
        
        consent = self._consents[consent_id]
        if consent.user_id != user_id:
            return False
        
        consent.status = ConsentStatus.DENIED
        
        return True
    
    def withdraw_consent(self, consent_id: str, user_id: str, 
                        reason: Optional[str] = None) -> bool:
        """
        Withdraw previously granted consent
        """
        if consent_id not in self._consents:
            return False
        
        consent = self._consents[consent_id]
        if consent.user_id != user_id or consent.status != ConsentStatus.GRANTED:
            return False
        
        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.now()
        if reason:
            consent.metadata['withdrawal_reason'] = reason
        
        return True
    
    def check_consent(self, user_id: str, purpose: ConsentPurpose, 
                     data_category: Optional[str] = None) -> bool:
        """
        Check if user has valid consent for specific purpose
        """
        if user_id not in self._user_consents:
            return False
        
        for consent_id in self._user_consents[user_id]:
            consent = self._consents[consent_id]
            
            if (consent.status == ConsentStatus.GRANTED and 
                consent.purpose == purpose and
                (consent.expires_at is None or consent.expires_at > datetime.now())):
                
                if data_category is None or data_category in consent.data_categories:
                    return True
        
        return False
    
    def get_user_consents(self, user_id: str) -> List[ConsentRecord]:
        """
        Get all consent records for a user
        """
        if user_id not in self._user_consents:
            return []
        
        return [self._consents[consent_id] for consent_id in self._user_consents[user_id]]
    
    def get_active_consents(self, user_id: str) -> List[ConsentRecord]:
        """
        Get currently active consents for a user
        """
        active_consents = []
        current_time = datetime.now()
        
        for consent in self.get_user_consents(user_id):
            if (consent.status == ConsentStatus.GRANTED and
                (consent.expires_at is None or consent.expires_at > current_time)):
                active_consents.append(consent)
        
        return active_consents
    
    def get_consent_history(self, user_id: str, purpose: Optional[ConsentPurpose] = None) -> List[Dict[str, Any]]:
        """
        Get consent history for audit purposes
        """
        history = []
        
        for consent in self.get_user_consents(user_id):
            if purpose is None or consent.purpose == purpose:
                history.append({
                    'consent_id': consent.consent_id,
                    'purpose': consent.purpose.value,
                    'status': consent.status.value,
                    'granted_at': consent.granted_at.isoformat(),
                    'withdrawn_at': consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    'expires_at': consent.expires_at.isoformat() if consent.expires_at else None,
                    'legal_basis': consent.legal_basis,
                    'data_categories': consent.data_categories,
                    'metadata': consent.metadata
                })
        
        return sorted(history, key=lambda x: x['granted_at'], reverse=True)
    
    def expire_consents(self) -> int:
        """
        Expire consents that have passed their expiration date
        """
        current_time = datetime.now()
        expired_count = 0
        
        for consent in self._consents.values():
            if (consent.status == ConsentStatus.GRANTED and 
                consent.expires_at and 
                consent.expires_at <= current_time):
                consent.status = ConsentStatus.EXPIRED
                expired_count += 1
        
        return expired_count
    
    def get_consent_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get summary of user's consent status
        """
        consents = self.get_user_consents(user_id)
        active_consents = self.get_active_consents(user_id)
        
        summary = {
            'user_id': user_id,
            'total_consents': len(consents),
            'active_consents': len(active_consents),
            'purposes': {},
            'data_categories': set(),
            'last_updated': None
        }
        
        for consent in consents:
            purpose_key = consent.purpose.value
            if purpose_key not in summary['purposes']:
                summary['purposes'][purpose_key] = {
                    'status': consent.status.value,
                    'granted_at': consent.granted_at.isoformat(),
                    'withdrawn_at': consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    'expires_at': consent.expires_at.isoformat() if consent.expires_at else None
                }
            
            summary['data_categories'].update(consent.data_categories)
            
            if summary['last_updated'] is None or consent.granted_at > datetime.fromisoformat(summary['last_updated']):
                summary['last_updated'] = consent.granted_at.isoformat()
        
        summary['data_categories'] = list(summary['data_categories'])
        
        return summary
    
    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all consent data for GDPR Article 20 (Right to data portability)
        """
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now().isoformat(),
            'consent_records': [
                {
                    'consent_id': consent.consent_id,
                    'purpose': consent.purpose.value,
                    'status': consent.status.value,
                    'granted_at': consent.granted_at.isoformat(),
                    'expires_at': consent.expires_at.isoformat() if consent.expires_at else None,
                    'withdrawn_at': consent.withdrawn_at.isoformat() if consent.withdrawn_at else None,
                    'legal_basis': consent.legal_basis,
                    'data_categories': consent.data_categories,
                    'processing_activities': consent.processing_activities,
                    'third_parties': consent.third_parties,
                    'metadata': consent.metadata
                }
                for consent in self.get_user_consents(user_id)
            ],
            'consent_summary': self.get_consent_summary(user_id)
        }
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all consent data for GDPR Article 17 (Right to erasure)
        """
        if user_id not in self._user_consents:
            return False
        
        # Remove all consent records
        consent_ids = self._user_consents[user_id].copy()
        for consent_id in consent_ids:
            del self._consents[consent_id]
        
        # Remove user from index
        del self._user_consents[user_id]
        
        return True
    
    def add_custom_template(self, template: ConsentTemplate) -> None:
        """Add a custom consent template"""
        self._templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[ConsentTemplate]:
        """Get consent template by ID"""
        return self._templates.get(template_id) or self._default_templates.get(template_id)
    
    def list_templates(self) -> List[ConsentTemplate]:
        """List all available consent templates"""
        all_templates = {}
        all_templates.update(self._default_templates)
        all_templates.update(self._templates)
        return list(all_templates.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consent management statistics"""
        total_consents = len(self._consents)
        status_counts = {}
        purpose_counts = {}
        
        for consent in self._consents.values():
            # Count by status
            status = consent.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by purpose
            purpose = consent.purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        
        return {
            'total_consents': total_consents,
            'total_users': len(self._user_consents),
            'status_distribution': status_counts,
            'purpose_distribution': purpose_counts,
            'active_consents': status_counts.get('granted', 0),
            'withdrawn_consents': status_counts.get('withdrawn', 0),
            'expired_consents': status_counts.get('expired', 0)
        }