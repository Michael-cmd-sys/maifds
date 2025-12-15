"""
GDPR Compliance Manager

Comprehensive GDPR compliance management for fraud detection systems:
- Article 15: Right of access
- Article 16: Right to rectification  
- Article 17: Right to erasure (Right to be forgotten)
- Article 18: Right to restriction of processing
- Article 20: Right to data portability
- Article 21: Right to object
- Article 25: Data protection by design and by default
- DPIA (Data Protection Impact Assessment) management
- Breach notification and reporting
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)

class GDPRRight(Enum):
    """GDPR rights that can be exercised"""
    RIGHT_OF_ACCESS = "right_of_access"  # Article 15
    RIGHT_OF_RECTIFICATION = "right_of_rectification"  # Article 16
    RIGHT_OF_ERASURE = "right_of_erasure"  # Article 17
    RIGHT_OF_RESTRICTION = "right_of_restriction"  # Article 18
    RIGHT_OF_DATA_PORTABILITY = "right_of_data_portability"  # Article 20
    RIGHT_TO_OBJECT = "right_to_object"  # Article 21

class DataRequestStatus(Enum):
    """Status of data subject requests"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXTENDED = "extended"

class BreachSeverity(Enum):
    """Data breach severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataSubjectRequest:
    """GDPR data subject request"""
    request_id: str
    user_id: str
    right: GDPRRight
    status: DataRequestStatus
    created_at: datetime
    due_date: datetime
    completed_at: Optional[datetime] = None
    details: str = ""
    response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DPIARecord:
    """Data Protection Impact Assessment record"""
    assessment_id: str
    system_name: str
    processing_purpose: str
    risk_level: str  # low, medium, high, critical
    assessment_date: datetime
    reviewer: str
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    mitigation_measures: List[str] = field(default_factory=list)
    approved: bool = False
    approval_date: Optional[datetime] = None

@dataclass
class DataBreachRecord:
    """Data breach notification record"""
    breach_id: str
    severity: BreachSeverity
    discovered_at: datetime
    reported_at: datetime
    contained_at: Optional[datetime] = None
    affected_users: int = 0
    data_types: List[str] = field(default_factory=list)
    description: str = ""
    impact_assessment: str = ""
    mitigation_actions: List[str] = field(default_factory=list)
    supervisory_authority_notified: bool = False
    individuals_notified: bool = False

class GDPRComplianceManager:
    """
    Comprehensive GDPR compliance management system
    """
    
    def __init__(self):
        self._requests: Dict[str, DataSubjectRequest] = {}
        self._user_requests: Dict[str, List[str]] = {}  # user_id -> request_ids
        self._dpias: Dict[str, DPIARecord] = {}
        self._breaches: Dict[str, DataBreachRecord] = {}
        self._legal_basis_registry: Dict[str, str] = {}
        self._processing_activities: Dict[str, Dict[str, Any]] = {}
        
        # Initialize legal basis for fraud detection
        self._initialize_legal_basis()
        
    def _initialize_legal_basis(self) -> None:
        """Initialize legal basis registry for fraud detection"""
        self._legal_basis_registry = {
            'fraud_detection': 'Legitimate interest in preventing and detecting fraud',
            'transaction_monitoring': 'Legal obligation under AML/CTF regulations',
            'risk_assessment': 'Legitimate interest in managing financial risk',
            'customer_verification': 'Performance of contract with customer',
            'security_monitoring': 'Legitimate interest in maintaining system security',
            'analytics': 'Legitimate interest in service improvement',
            'marketing': 'Explicit consent from data subject'
        }
        
        self._processing_activities = {
            'fraud_detection': {
                'purpose': 'Prevent and detect fraudulent activities',
                'data_categories': ['transaction_data', 'device_info', 'behavioral_patterns'],
                'retention_period': '7 years',
                'third_parties': ['payment_processors', 'credit_bureaus', 'law_enforcement'],
                'international_transfers': True,
                'automated_decision_making': True,
                'profiling': True
            },
            'customer_verification': {
                'purpose': 'Verify customer identity for service provision',
                'data_categories': ['personal_identifiers', 'documents', 'biometric_data'],
                'retention_period': '5 years after relationship ends',
                'third_parties': ['identity_verification_services'],
                'international_transfers': False,
                'automated_decision_making': False,
                'profiling': False
            }
        }
    
    def create_data_subject_request(self, user_id: str, right: GDPRRight, 
                                  details: str = "") -> str:
        """
        Create a new data subject request
        """
        request_id = str(uuid.uuid4())
        
        # Standard response time is 1 month, can be extended to 2 months for complex requests
        due_date = datetime.now() + timedelta(days=30)
        
        request = DataSubjectRequest(
            request_id=request_id,
            user_id=user_id,
            right=right,
            status=DataRequestStatus.PENDING,
            created_at=datetime.now(),
            due_date=due_date,
            details=details
        )
        
        self._requests[request_id] = request
        
        if user_id not in self._user_requests:
            self._user_requests[user_id] = []
        self._user_requests[user_id].append(request_id)
        
        logger.info(f"Created GDPR request {request_id} for user {user_id}: {right.value}")
        
        return request_id
    
    def process_access_request(self, request_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Process right of access request (Article 15)
        """
        if request_id not in self._requests:
            return False
        
        request = self._requests[request_id]
        if request.right != GDPRRight.RIGHT_OF_ACCESS:
            return False
        
        request.status = DataRequestStatus.PROCESSING
        
        # Compile all personal data
        access_data = {
            'personal_data': user_data,
            'processing_activities': self._get_user_processing_activities(request.user_id),
            'legal_basis': self._get_user_legal_basis(request.user_id),
            'data_recipients': self._get_data_recipients(request.user_id),
            'retention_periods': self._get_retention_periods(request.user_id),
            'automated_decisions': self._get_automated_decisions(request.user_id),
            'international_transfers': self._get_international_transfers(request.user_id)
        }
        
        request.response = json.dumps(access_data, indent=2, default=str)
        request.status = DataRequestStatus.COMPLETED
        request.completed_at = datetime.now()
        
        logger.info(f"Completed access request {request_id} for user {request.user_id}")
        
        return True
    
    def process_erasure_request(self, request_id: str, deletion_callback) -> bool:
        """
        Process right to erasure request (Article 17)
        """
        if request_id not in self._requests:
            return False
        
        request = self._requests[request_id]
        if request.right != GDPRRight.RIGHT_OF_ERASURE:
            return False
        
        request.status = DataRequestStatus.PROCESSING
        
        try:
            # Call deletion callback to actually delete user data
            deletion_result = deletion_callback(request.user_id)
            
            if deletion_result:
                request.response = "All personal data has been successfully deleted"
                request.status = DataRequestStatus.COMPLETED
                request.completed_at = datetime.now()
                
                logger.info(f"Completed erasure request {request_id} for user {request.user_id}")
                return True
            else:
                request.response = "Unable to complete deletion due to legal obligations"
                request.status = DataRequestStatus.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"Error processing erasure request {request_id}: {str(e)}")
            request.response = f"Error during deletion: {str(e)}"
            request.status = DataRequestStatus.REJECTED
            return False
    
    def process_portability_request(self, request_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Process right to data portability request (Article 20)
        """
        if request_id not in self._requests:
            return False
        
        request = self._requests[request_id]
        if request.right != GDPRRight.RIGHT_OF_DATA_PORTABILITY:
            return False
        
        request.status = DataRequestStatus.PROCESSING
        
        # Compile portable data in machine-readable format
        portable_data = {
            'user_id': request.user_id,
            'export_date': datetime.now().isoformat(),
            'personal_data': user_data,
            'processing_purposes': self._get_user_processing_activities(request.user_id),
            'data_categories': self._get_user_data_categories(request.user_id)
        }
        
        request.response = json.dumps(portable_data, indent=2, default=str)
        request.status = DataRequestStatus.COMPLETED
        request.completed_at = datetime.now()
        
        logger.info(f"Completed portability request {request_id} for user {request.user_id}")
        
        return True
    
    def create_dpia(self, system_name: str, processing_purpose: str, 
                   risk_level: str, reviewer: str) -> str:
        """
        Create Data Protection Impact Assessment
        """
        assessment_id = str(uuid.uuid4())
        
        dpia = DPIARecord(
            assessment_id=assessment_id,
            system_name=system_name,
            processing_purpose=processing_purpose,
            risk_level=risk_level,
            assessment_date=datetime.now(),
            reviewer=reviewer
        )
        
        self._dpias[assessment_id] = dpia
        
        logger.info(f"Created DPIA {assessment_id} for {system_name}")
        
        return assessment_id
    
    def update_dpia_findings(self, assessment_id: str, findings: List[str], 
                           recommendations: List[str], 
                           mitigation_measures: List[str]) -> bool:
        """Update DPIA with findings and recommendations"""
        if assessment_id not in self._dpias:
            return False
        
        dpia = self._dpias[assessment_id]
        dpia.findings = findings
        dpia.recommendations = recommendations
        dpia.mitigation_measures = mitigation_measures
        
        return True
    
    def approve_dpia(self, assessment_id: str, approver: str) -> bool:
        """Approve DPIA assessment"""
        if assessment_id not in self._dpias:
            return False
        
        dpia = self._dpias[assessment_id]
        dpia.approved = True
        dpia.approval_date = datetime.now()
        dpia.reviewer = approver
        
        logger.info(f"Approved DPIA {assessment_id} by {approver}")
        
        return True
    
    def report_data_breach(self, severity: BreachSeverity, affected_users: int,
                         data_types: List[str], description: str) -> str:
        """
        Report a data breach
        """
        breach_id = str(uuid.uuid4())
        
        breach = DataBreachRecord(
            breach_id=breach_id,
            severity=severity,
            discovered_at=datetime.now(),
            reported_at=datetime.now(),
            affected_users=affected_users,
            data_types=data_types,
            description=description
        )
        
        self._breaches[breach_id] = breach
        
        # Check if supervisory authority needs to be notified (within 72 hours for high/critical)
        if severity in [BreachSeverity.HIGH, BreachSeverity.CRITICAL]:
            breach.supervisory_authority_notified = True
            logger.warning(f"High severity breach {breach_id} - supervisory authority notification required")
        
        logger.warning(f"Reported data breach {breach_id}: {severity.value}, {affected_users} users affected")
        
        return breach_id
    
    def contain_breach(self, breach_id: str, mitigation_actions: List[str]) -> bool:
        """Mark breach as contained with mitigation actions"""
        if breach_id not in self._breaches:
            return False
        
        breach = self._breaches[breach_id]
        breach.contained_at = datetime.now()
        breach.mitigation_actions = mitigation_actions
        
        logger.info(f"Contained breach {breach_id}")
        
        return True
    
    def get_pending_requests(self) -> List[DataSubjectRequest]:
        """Get all pending data subject requests"""
        return [req for req in self._requests.values() 
                if req.status == DataRequestStatus.PENDING]
    
    def get_overdue_requests(self) -> List[DataSubjectRequest]:
        """Get overdue data subject requests"""
        now = datetime.now()
        return [req for req in self._requests.values() 
                if req.due_date < now and req.status not in [DataRequestStatus.COMPLETED, DataRequestStatus.REJECTED]]
    
    def get_user_requests(self, user_id: str) -> List[DataSubjectRequest]:
        """Get all requests for a specific user"""
        if user_id not in self._user_requests:
            return []
        
        return [self._requests[req_id] for req_id in self._user_requests[user_id]]
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        now = datetime.now()
        
        # Request statistics
        total_requests = len(self._requests)
        completed_requests = len([r for r in self._requests.values() if r.status == DataRequestStatus.COMPLETED])
        pending_requests = len([r for r in self._requests.values() if r.status == DataRequestStatus.PENDING])
        overdue_requests = len(self.get_overdue_requests())
        
        # DPIA statistics
        total_dpias = len(self._dpias)
        approved_dpias = len([d for d in self._dpias.values() if d.approved])
        high_risk_dpias = len([d for d in self._dpias.values() if d.risk_level in ['high', 'critical']])
        
        # Breach statistics
        total_breaches = len(self._breaches)
        critical_breaches = len([b for b in self._breaches.values() if b.severity == BreachSeverity.CRITICAL])
        uncontained_breaches = len([b for b in self._breaches.values() if b.contained_at is None])
        
        return {
            'report_date': now.isoformat(),
            'data_subject_requests': {
                'total': total_requests,
                'completed': completed_requests,
                'pending': pending_requests,
                'overdue': overdue_requests,
                'completion_rate': completed_requests / total_requests if total_requests > 0 else 0
            },
            'dpia_assessments': {
                'total': total_dpias,
                'approved': approved_dpias,
                'high_risk': high_risk_dpias,
                'approval_rate': approved_dpias / total_dpias if total_dpias > 0 else 0
            },
            'data_breaches': {
                'total': total_breaches,
                'critical': critical_breaches,
                'uncontained': uncontained_breaches,
                'containment_rate': (total_breaches - uncontained_breaches) / total_breaches if total_breaches > 0 else 0
            },
            'processing_activities': len(self._processing_activities),
            'legal_basis_registered': len(self._legal_basis_registry)
        }
    
    def _get_user_processing_activities(self, user_id: str) -> List[str]:
        """Get processing activities for user"""
        return list(self._processing_activities.keys())
    
    def _get_user_legal_basis(self, user_id: str) -> Dict[str, str]:
        """Get legal basis for user data processing"""
        return self._legal_basis_registry
    
    def _get_data_recipients(self, user_id: str) -> List[str]:
        """Get data recipients for user data"""
        recipients = set()
        for activity in self._processing_activities.values():
            recipients.update(activity.get('third_parties', []))
        return list(recipients)
    
    def _get_retention_periods(self, user_id: str) -> Dict[str, str]:
        """Get retention periods for user data"""
        return {activity: details['retention_period'] 
                for activity, details in self._processing_activities.items()}
    
    def _get_automated_decisions(self, user_id: str) -> List[str]:
        """Get automated decision making processes"""
        return [activity for activity, details in self._processing_activities.items()
                if details.get('automated_decision_making', False)]
    
    def _get_international_transfers(self, user_id: str) -> List[str]:
        """Get international data transfers"""
        return [activity for activity, details in self._processing_activities.items()
                if details.get('international_transfers', False)]
    
    def _get_user_data_categories(self, user_id: str) -> List[str]:
        """Get data categories for user"""
        categories = set()
        for activity in self._processing_activities.values():
            categories.update(activity.get('data_categories', []))
        return list(categories)