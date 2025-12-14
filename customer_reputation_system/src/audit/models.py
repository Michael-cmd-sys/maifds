"""
Pydantic models for audit trail and compliance
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class EventType(str, Enum):
    """Types of audit events"""
    DECISION_MADE = "decision_made"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    MODEL_PREDICTION = "model_prediction"
    USER_LOGIN = "user_login"
    CONFIG_CHANGE = "config_change"
    SYSTEM_ERROR = "system_error"
    PRIVACY_REQUEST = "privacy_request"


class ComponentType(str, Enum):
    """System components that generate events"""
    CUSTOMER_REPUTATION = "customer_reputation"
    MEL_DEV_CALL_DEFENSE = "mel_dev_call_defense"
    MEL_DEV_CLICK_TX = "mel_dev_click_tx"
    MEL_DEV_PROACTIVE_WARNING = "mel_dev_proactive_warning"
    HUAWEI_PHISHING_DETECTOR = "huawei_phishing_detector"
    HUAWEI_BLACKLIST = "huawei_blacklist"
    HUAWEI_PROACTIVE_WARNING = "huawei_proactive_warning"
    AUDIT_SERVICE = "audit_service"
    PRIVACY_SERVICE = "privacy_service"


class PrivacyImpact(str, Enum):
    """Privacy impact levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Core audit event model"""
    
    event_id: str = Field(description="Unique event identifier")
    timestamp: datetime = Field(description="Event timestamp")
    event_type: EventType = Field(description="Type of event")
    component: ComponentType = Field(description="System component")
    user_id: Optional[str] = Field(None, description="User who triggered event")
    session_id: Optional[str] = Field(None, description="Session identifier")
    entity_id: Optional[str] = Field(None, description="Primary entity ID")
    entity_type: Optional[str] = Field(None, description="Type of entity")
    decision_data: Optional[Dict[str, Any]] = Field(None, description="Decision-related data")
    explanation_data: Optional[Dict[str, Any]] = Field(None, description="Explanation data")
    privacy_impact: PrivacyImpact = Field(PrivacyImpact.NONE, description="Privacy impact level")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    success: bool = Field(True, description="Whether operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    model_config = {"extra": "forbid"}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.model_dump()
        data["timestamp"] = self.timestamp.isoformat()
        # Convert enums to strings
        data["event_type"] = self.event_type.value
        data["component"] = self.component.value
        data["privacy_impact"] = self.privacy_impact.value
        return data


class DecisionEvent(BaseModel):
    """Specialized model for decision events"""
    
    event_id: str
    timestamp: datetime
    component: ComponentType
    decision_type: str = Field(description="Type of decision made")
    entity_id: str
    entity_type: str
    risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    decision_outcome: str = Field(description="Final decision")
    decision_factors: Optional[Dict[str, float]] = Field(None, description="Factors influencing decision")
    rule_triggers: Optional[List[str]] = Field(None, description="Rules that were triggered")
    model_predictions: Optional[Dict[str, Any]] = Field(None, description="ML model predictions")
    explanation_summary: Optional[str] = Field(None, description="Brief explanation")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    model_config = {"extra": "forbid"}
    
    def to_audit_event(self) -> AuditEvent:
        """Convert to standard audit event"""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            event_type=EventType.DECISION_MADE,
            component=self.component,
            user_id=self.user_id,
            session_id=self.session_id,
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            decision_data={
                "decision_type": self.decision_type,
                "risk_score": self.risk_score,
                "confidence_score": self.confidence_score,
                "decision_outcome": self.decision_outcome,
                "decision_factors": self.decision_factors,
                "rule_triggers": self.rule_triggers,
                "model_predictions": self.model_predictions
            },
            explanation_data={
                "explanation_summary": self.explanation_summary,
                "decision_factors": self.decision_factors,
                "rule_triggers": self.rule_triggers
            },
            privacy_impact=PrivacyImpact.MEDIUM if self.entity_type == "user" else PrivacyImpact.LOW,
            ip_address=self.ip_address,
            user_agent=None,
            error_message=None,
            metadata=None,
            success=True
        )


class DataAccessEvent(BaseModel):
    """Specialized model for data access events"""
    
    event_id: str
    timestamp: datetime
    component: ComponentType
    access_type: str = Field(description="Type of access: read, write, delete")
    data_type: str = Field(description="Type of data accessed")
    record_ids: Optional[List[str]] = Field(None, description="IDs of records accessed")
    query_params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    result_count: Optional[int] = Field(None, description="Number of results")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    purpose: Optional[str] = Field(None, description="Purpose of access")
    
    model_config = {"extra": "forbid"}
    
    def to_audit_event(self) -> AuditEvent:
        """Convert to standard audit event"""
        privacy_impact = PrivacyImpact.HIGH if self.access_type in ["read", "write"] else PrivacyImpact.CRITICAL
        
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            event_type=EventType.DATA_ACCESS,
            component=self.component,
            user_id=self.user_id,
            session_id=self.session_id,
            entity_id=self.record_ids[0] if self.record_ids else None,
            entity_type=self.data_type,
            decision_data={
                "access_type": self.access_type,
                "data_type": self.data_type,
                "record_ids": self.record_ids,
                "query_params": self.query_params,
                "result_count": self.result_count,
                "purpose": self.purpose
            },
            explanation_data=None,
            privacy_impact=privacy_impact,
            ip_address=self.ip_address,
            user_agent=None,
            error_message=None,
            metadata=None,
            success=True
        )


class ModelPredictionEvent(BaseModel):
    """Specialized model for ML model predictions"""
    
    event_id: str
    timestamp: datetime
    component: ComponentType
    model_name: str = Field(description="Name of the model")
    model_version: Optional[str] = Field(None, description="Model version")
    input_features: Dict[str, Any] = Field(description="Input features")
    prediction: Any = Field(description="Model prediction")
    prediction_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    explanation_method: Optional[str] = Field(None, description="Method used for explanation")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    entity_id: Optional[str] = None
    
    model_config = {"extra": "forbid"}
    
    def to_audit_event(self) -> AuditEvent:
        """Convert to standard audit event"""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            event_type=EventType.MODEL_PREDICTION,
            component=self.component,
            user_id=self.user_id,
            session_id=self.session_id,
            entity_id=self.entity_id,
            entity_type="model_prediction",
            decision_data={
                "model_name": self.model_name,
                "model_version": self.model_version,
                "prediction": self.prediction,
                "prediction_probability": self.prediction_probability,
                "processing_time_ms": self.processing_time_ms
            },
            explanation_data={
                "input_features": self.input_features,
                "feature_importance": self.feature_importance,
                "explanation_method": self.explanation_method
            },
            privacy_impact=PrivacyImpact.LOW,
            ip_address=None,
            user_agent=None,
            error_message=None,
            metadata=None,
            success=True
        )


class PrivacyRequestEvent(BaseModel):
    """Specialized model for privacy-related requests"""
    
    event_id: str
    timestamp: datetime
    component: ComponentType
    request_type: str = Field(description="Type of privacy request")
    user_id: str = Field(description="User making request")
    request_data: Optional[Dict[str, Any]] = Field(None, description="Request-specific data")
    processing_status: str = Field(description="Status of request processing")
    completion_timestamp: Optional[datetime] = Field(None, description="When request was completed")
    data_deleted_count: Optional[int] = Field(None, description="Number of records deleted")
    data_exported_count: Optional[int] = Field(None, description="Number of records exported")
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    model_config = {"extra": "forbid"}
    
    def to_audit_event(self) -> AuditEvent:
        """Convert to standard audit event"""
        return AuditEvent(
            event_id=self.event_id,
            timestamp=self.timestamp,
            event_type=EventType.PRIVACY_REQUEST,
            component=self.component,
            user_id=self.user_id,
            session_id=None,
            entity_id=self.user_id,
            entity_type="privacy_request",
            decision_data={
                "request_type": self.request_type,
                "processing_status": self.processing_status,
                "data_deleted_count": self.data_deleted_count,
                "data_exported_count": self.data_exported_count
            },
            explanation_data={
                "request_data": self.request_data
            },
            privacy_impact=PrivacyImpact.CRITICAL,
            ip_address=self.ip_address,
            user_agent=self.user_agent,
            error_message=None,
            metadata=None,
            success=self.processing_status == "completed"
        )