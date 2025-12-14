"""
Privacy Access Control System

Role-based access control for privacy-sensitive data:
- Attribute-based access control (ABAC)
- Data classification-based access
- Purpose limitation enforcement
- Need-to-know principle implementation
- Audit logging for access attempts
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import logging

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles in the system"""
    ANALYST = "analyst"
    INVESTIGATOR = "investigator"
    COMPLIANCE_OFFICER = "compliance_officer"
    SYSTEM_ADMIN = "system_admin"
    DATA_PROTECTION_OFFICER = "dpo"
    AUDITOR = "auditor"
    SERVICE_ACCOUNT = "service_account"

class DataSensitivity(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CRITICAL = "critical"

class AccessPurpose(Enum):
    """Purposes for data access"""
    FRAUD_INVESTIGATION = "fraud_investigation"
    COMPLIENCE_AUDIT = "compliance_audit"
    SYSTEM_MAINTENANCE = "system_maintenance"
    DATA_ANALYSIS = "data_analysis"
    LEGAL_PROCEEDING = "legal_proceeding"
    REGULATORY_REPORTING = "regulatory_reporting"
    SECURITY_MONITORING = "security_monitoring"

@dataclass
class AccessPolicy:
    """Access control policy"""
    policy_id: str
    name: str
    description: str
    roles: List[UserRole]
    data_sensitivity: List[DataSensitivity]
    purposes: List[AccessPurpose]
    allowed_operations: List[str]  # read, write, delete, export
    time_restrictions: Optional[Dict[str, Any]] = None
    location_restrictions: Optional[List[str]] = None
    approval_required: bool = False
    approvers: List[str] = field(default_factory=list)
    active: bool = True

@dataclass
class AccessRequest:
    """Data access request"""
    request_id: str
    user_id: str
    role: UserRole
    purpose: AccessPurpose
    data_type: str
    sensitivity: DataSensitivity
    operation: str
    requested_at: datetime
    expires_at: Optional[datetime] = None
    approved: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    justification: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessLog:
    """Access attempt log"""
    log_id: str
    user_id: str
    role: UserRole
    resource: str
    operation: str
    purpose: AccessPurpose
    granted: bool
    timestamp: datetime
    ip_address: str
    user_agent: str
    session_id: str
    justification: str = ""
    policy_id: Optional[str] = None

class PrivacyAccessController:
    """
    Comprehensive privacy access control system
    """
    
    def __init__(self):
        self._policies: Dict[str, AccessPolicy] = {}
        self._requests: Dict[str, AccessRequest] = {}
        self._access_logs: List[AccessLog] = []
        self._user_sessions: Dict[str, Dict[str, Any]] = {}
        self._blocked_ips: Set[str] = set()
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self) -> None:
        """Initialize default access control policies"""
        
        # Fraud investigation policy
        self._policies['fraud_investigation'] = AccessPolicy(
            policy_id='fraud_investigation',
            name='Fraud Investigation Access',
            description='Access for fraud investigation activities',
            roles=[UserRole.ANALYST, UserRole.INVESTIGATOR],
            data_sensitivity=[DataSensitivity.INTERNAL, DataSensitivity.CONFIDENTIAL],
            purposes=[AccessPurpose.FRAUD_INVESTIGATION],
            allowed_operations=['read'],
            time_restrictions={'business_hours_only': True},
            approval_required=False
        )
        
        # Compliance audit policy
        self._policies['compliance_audit'] = AccessPolicy(
            policy_id='compliance_audit',
            name='Compliance Audit Access',
            description='Access for compliance and audit activities',
            roles=[UserRole.COMPLIANCE_OFFICER, UserRole.AUDITOR, UserRole.DATA_PROTECTION_OFFICER],
            data_sensitivity=[DataSensitivity.INTERNAL, DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED],
            purposes=[AccessPurpose.COMPLIENCE_AUDIT, AccessPurpose.REGULATORY_REPORTING],
            allowed_operations=['read', 'export'],
            approval_required=False
        )
        
        # System administration policy
        self._policies['system_admin'] = AccessPolicy(
            policy_id='system_admin',
            name='System Administration Access',
            description='Full system access for maintenance',
            roles=[UserRole.SYSTEM_ADMIN],
            data_sensitivity=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL, DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED, DataSensitivity.CRITICAL],
            purposes=[AccessPurpose.SYSTEM_MAINTENANCE, AccessPurpose.SECURITY_MONITORING],
            allowed_operations=['read', 'write', 'delete'],
            approval_required=False
        )
        
        # Legal proceeding policy
        self._policies['legal_proceeding'] = AccessPolicy(
            policy_id='legal_proceeding',
            name='Legal Proceeding Access',
            description='Access for legal proceedings and investigations',
            roles=[UserRole.COMPLIANCE_OFFICER, UserRole.INVESTIGATOR],
            data_sensitivity=[DataSensitivity.CONFIDENTIAL, DataSensitivity.RESTRICTED],
            purposes=[AccessPurpose.LEGAL_PROCEEDING],
            allowed_operations=['read', 'export'],
            approval_required=True,
            approvers=['legal_counsel', 'dpo']
        )
        
        # Data analysis policy
        self._policies['data_analysis'] = AccessPolicy(
            policy_id='data_analysis',
            name='Data Analysis Access',
            description='Access for data analysis and research',
            roles=[UserRole.ANALYST],
            data_sensitivity=[DataSensitivity.PUBLIC, DataSensitivity.INTERNAL],
            purposes=[AccessPurpose.DATA_ANALYSIS],
            allowed_operations=['read'],
            approval_required=True,
            approvers=['data_governance_officer']
        )
    
    def check_access(self, user_id: str, role: UserRole, resource: str,
                    operation: str, purpose: AccessPurpose, 
                    sensitivity: DataSensitivity,
                    context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Check if user has access to resource
        """
        context = context or {}
        current_time = datetime.now()
        
        # Check if IP is blocked
        ip_address = context.get('ip_address', '')
        if ip_address in self._blocked_ips:
            self._log_access(user_id, role, resource, operation, purpose, False, context)
            return False, "IP address blocked"
        
        # Find applicable policies
        applicable_policies = []
        for policy in self._policies.values():
            if (policy.active and
                role in policy.roles and
                sensitivity in policy.data_sensitivity and
                purpose in policy.purposes and
                operation in policy.allowed_operations):
                applicable_policies.append(policy)
        
        if not applicable_policies:
            self._log_access(user_id, role, resource, operation, purpose, False, context)
            return False, "No applicable policy found"
        
        # Check time restrictions
        for policy in applicable_policies:
            if policy.time_restrictions:
                if not self._check_time_restrictions(policy.time_restrictions, current_time):
                    self._log_access(user_id, role, resource, operation, purpose, False, context, policy.policy_id)
                    return False, "Access outside allowed time window"
        
        # Check location restrictions
        for policy in applicable_policies:
            if policy.location_restrictions:
                user_location = context.get('location', '')
                if user_location not in policy.location_restrictions:
                    self._log_access(user_id, role, resource, operation, purpose, False, context, policy.policy_id)
                    return False, "Access from restricted location"
        
        # Check if approval is required
        for policy in applicable_policies:
            if policy.approval_required:
                approved_request = self._check_approved_request(user_id, resource, operation, purpose)
                if not approved_request:
                    self._log_access(user_id, role, resource, operation, purpose, False, context, policy.policy_id)
                    return False, "Approval required"
        
        # Access granted
        policy_id = applicable_policies[0].policy_id
        self._log_access(user_id, role, resource, operation, purpose, True, context, policy_id)
        
        return True, None
    
    def request_access(self, user_id: str, role: UserRole, resource: str,
                      operation: str, purpose: AccessPurpose,
                      sensitivity: DataSensitivity, justification: str,
                      expires_hours: Optional[int] = None) -> str:
        """
        Create access request for approval
        """
        request_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_hours:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        request = AccessRequest(
            request_id=request_id,
            user_id=user_id,
            role=role,
            purpose=purpose,
            data_type=resource,
            sensitivity=sensitivity,
            operation=operation,
            requested_at=datetime.now(),
            expires_at=expires_at,
            justification=justification
        )
        
        self._requests[request_id] = request
        
        logger.info(f"Created access request {request_id} for user {user_id}")
        
        return request_id
    
    def approve_request(self, request_id: str, approver_id: str) -> bool:
        """
        Approve an access request
        """
        if request_id not in self._requests:
            return False
        
        request = self._requests[request_id]
        if request.approved:
            return False
        
        request.approved = True
        request.approved_by = approver_id
        request.approved_at = datetime.now()
        
        logger.info(f"Approved access request {request_id} by {approver_id}")
        
        return True
    
    def deny_request(self, request_id: str, approver_id: str, reason: str) -> bool:
        """
        Deny an access request
        """
        if request_id not in self._requests:
            return False
        
        request = self._requests[request_id]
        if request.approved:
            return False
        
        request.metadata['denied_by'] = approver_id
        request.metadata['denied_at'] = datetime.now().isoformat()
        request.metadata['denial_reason'] = reason
        
        logger.info(f"Denied access request {request_id} by {approver_id}: {reason}")
        
        return True
    
    def revoke_access(self, user_id: str, resource: str, reason: str) -> bool:
        """
        Revoke user's access to a resource
        """
        # Mark all approved requests for this user/resource as expired
        revoked_count = 0
        for request in self._requests.values():
            if (request.user_id == user_id and 
                request.data_type == resource and 
                request.approved and
                request.expires_at and
                request.expires_at > datetime.now()):
                
                request.expires_at = datetime.now()
                request.metadata['revoked'] = True
                request.metadata['revocation_reason'] = reason
                revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} access grants for user {user_id} to {resource}")
        
        return revoked_count > 0
    
    def block_ip(self, ip_address: str, reason: str) -> bool:
        """Block an IP address from accessing the system"""
        self._blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP address {ip_address}: {reason}")
        return True
    
    def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address"""
        if ip_address in self._blocked_ips:
            self._blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP address {ip_address}")
            return True
        return False
    
    def get_pending_requests(self) -> List[AccessRequest]:
        """Get all pending access requests"""
        return [req for req in self._requests.values() if not req.approved]
    
    def get_user_requests(self, user_id: str) -> List[AccessRequest]:
        """Get all requests for a specific user"""
        return [req for req in self._requests.values() if req.user_id == user_id]
    
    def get_access_logs(self, user_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       granted_only: bool = False) -> List[AccessLog]:
        """
        Get access logs with filtering
        """
        logs = self._access_logs
        
        # Filter by user
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        # Filter by time range
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        # Filter by granted status
        if granted_only:
            logs = [log for log in logs if log.granted]
        
        return logs
    
    def get_access_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        total_logs = len(self._access_logs)
        granted_logs = len([log for log in self._access_logs if log.granted])
        denied_logs = total_logs - granted_logs
        
        # Statistics by role
        role_stats = {}
        for log in self._access_logs:
            role = log.role.value
            if role not in role_stats:
                role_stats[role] = {'total': 0, 'granted': 0, 'denied': 0}
            role_stats[role]['total'] += 1
            if log.granted:
                role_stats[role]['granted'] += 1
            else:
                role_stats[role]['denied'] += 1
        
        # Statistics by purpose
        purpose_stats = {}
        for log in self._access_logs:
            purpose = log.purpose.value
            if purpose not in purpose_stats:
                purpose_stats[purpose] = {'total': 0, 'granted': 0, 'denied': 0}
            purpose_stats[purpose]['total'] += 1
            if log.granted:
                purpose_stats[purpose]['granted'] += 1
            else:
                purpose_stats[purpose]['denied'] += 1
        
        return {
            'total_access_attempts': total_logs,
            'granted_access': granted_logs,
            'denied_access': denied_logs,
            'approval_rate': granted_logs / total_logs if total_logs > 0 else 0,
            'pending_requests': len(self.get_pending_requests()),
            'blocked_ips': len(self._blocked_ips),
            'active_policies': len([p for p in self._policies.values() if p.active]),
            'statistics_by_role': role_stats,
            'statistics_by_purpose': purpose_stats
        }
    
    def _check_time_restrictions(self, restrictions: Dict[str, Any], 
                                current_time: datetime) -> bool:
        """Check if current time meets time restrictions"""
        if restrictions.get('business_hours_only', False):
            # Business hours: 9 AM - 6 PM, Monday - Friday
            if current_time.weekday() >= 5:  # Weekend
                return False
            if not (9 <= current_time.hour < 18):
                return False
        
        if 'start_hour' in restrictions and 'end_hour' in restrictions:
            if not (restrictions['start_hour'] <= current_time.hour < restrictions['end_hour']):
                return False
        
        return True
    
    def _check_approved_request(self, user_id: str, resource: str,
                               operation: str, purpose: AccessPurpose) -> bool:
        """Check if user has an approved request for this access"""
        for request in self._requests.values():
            if (request.user_id == user_id and
                request.data_type == resource and
                request.operation == operation and
                request.purpose == purpose and
                request.approved and
                (request.expires_at is None or request.expires_at > datetime.now())):
                return True
        return False
    
    def _log_access(self, user_id: str, role: UserRole, resource: str,
                   operation: str, purpose: AccessPurpose, granted: bool,
                   context: Dict[str, Any], policy_id: Optional[str] = None) -> None:
        """Log access attempt"""
        log = AccessLog(
            log_id=str(uuid.uuid4()),
            user_id=user_id,
            role=role,
            resource=resource,
            operation=operation,
            purpose=purpose,
            granted=granted,
            timestamp=datetime.now(),
            ip_address=context.get('ip_address', ''),
            user_agent=context.get('user_agent', ''),
            session_id=context.get('session_id', ''),
            justification=context.get('justification', ''),
            policy_id=policy_id
        )
        
        self._access_logs.append(log)
        
        # Keep only last 10000 logs to prevent memory issues
        if len(self._access_logs) > 10000:
            self._access_logs = self._access_logs[-10000:]
    
    def add_policy(self, policy: AccessPolicy) -> None:
        """Add a new access policy"""
        self._policies[policy.policy_id] = policy
    
    def update_policy(self, policy_id: str, **kwargs) -> bool:
        """Update an existing policy"""
        if policy_id not in self._policies:
            return False
        
        policy = self._policies[policy_id]
        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
        
        return True
    
    def deactivate_policy(self, policy_id: str) -> bool:
        """Deactivate a policy"""
        if policy_id not in self._policies:
            return False
        
        self._policies[policy_id].active = False
        return True
    
    def get_policy(self, policy_id: str) -> Optional[AccessPolicy]:
        """Get policy by ID"""
        return self._policies.get(policy_id)
    
    def list_policies(self) -> List[AccessPolicy]:
        """List all policies"""
        return list(self._policies.values())