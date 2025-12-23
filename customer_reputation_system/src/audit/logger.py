"""
Enhanced audit logging service for comprehensive system monitoring
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from customer_reputation_system.config.logging_config import setup_logger
from . import models as audit_models
from . import database as audit_db

# Import specific classes
from audit_models import (
    AuditEvent, DecisionEvent, DataAccessEvent, 
    ModelPredictionEvent, PrivacyRequestEvent,
    EventType, ComponentType, PrivacyImpact
)
from audit_db import AuditDatabaseManager

logger = setup_logger(__name__)


class AuditLogger:
    """Centralized audit logging service"""

    def __init__(self, audit_db: Optional[AuditDatabaseManager] = None):
        """Initialize audit logger"""
        self.audit_db = audit_db or AuditDatabaseManager()
        logger.info("AuditLogger initialized")

    def log_decision(
        self,
        decision_type: str,
        entity_id: str,
        entity_type: str,
        risk_score: Optional[float] = None,
        confidence_score: Optional[float] = None,
        decision_outcome: str = "",
        decision_factors: Optional[Dict[str, float]] = None,
        rule_triggers: Optional[list] = None,
        model_predictions: Optional[Dict[str, Any]] = None,
        explanation_summary: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        component: ComponentType = ComponentType.CUSTOMER_REPUTATION
    ) -> bool:
        """Log a decision event"""
        try:
            event_id = str(uuid.uuid4())
            
            decision_event = DecisionEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                component=component,
                decision_type=decision_type,
                entity_id=entity_id,
                entity_type=entity_type,
                risk_score=risk_score,
                confidence_score=confidence_score,
                decision_outcome=decision_outcome,
                decision_factors=decision_factors,
                rule_triggers=rule_triggers,
                model_predictions=model_predictions,
                explanation_summary=explanation_summary,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address
            )
            
            return self.audit_db.store_decision_event(decision_event)
            
        except Exception as e:
            logger.error(f"Failed to log decision event: {e}")
            return False

    def log_data_access(
        self,
        access_type: str,
        data_type: str,
        record_ids: Optional[list] = None,
        query_params: Optional[Dict[str, Any]] = None,
        result_count: Optional[int] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        purpose: Optional[str] = None,
        component: ComponentType = ComponentType.CUSTOMER_REPUTATION
    ) -> bool:
        """Log a data access event"""
        try:
            event_id = str(uuid.uuid4())
            
            access_event = DataAccessEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                component=component,
                access_type=access_type,
                data_type=data_type,
                record_ids=record_ids,
                query_params=query_params,
                result_count=result_count,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                purpose=purpose
            )
            
            return self.audit_db.store_data_access_event(access_event)
            
        except Exception as e:
            logger.error(f"Failed to log data access event: {e}")
            return False

    def log_model_prediction(
        self,
        model_name: str,
        input_features: Dict[str, Any],
        prediction: Any,
        prediction_probability: Optional[float] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        explanation_method: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        model_version: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        component: ComponentType = ComponentType.CUSTOMER_REPUTATION
    ) -> bool:
        """Log a model prediction event"""
        try:
            event_id = str(uuid.uuid4())
            
            prediction_event = ModelPredictionEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                component=component,
                model_name=model_name,
                model_version=model_version,
                input_features=input_features,
                prediction=prediction,
                prediction_probability=prediction_probability,
                feature_importance=feature_importance,
                explanation_method=explanation_method,
                processing_time_ms=processing_time_ms,
                user_id=user_id,
                session_id=session_id,
                entity_id=entity_id
            )
            
            return self.audit_db.store_model_prediction_event(prediction_event)
            
        except Exception as e:
            logger.error(f"Failed to log model prediction event: {e}")
            return False

    def log_privacy_request(
        self,
        request_type: str,
        user_id: str,
        request_data: Optional[Dict[str, Any]] = None,
        processing_status: str = "pending",
        completion_timestamp: Optional[datetime] = None,
        data_deleted_count: Optional[int] = None,
        data_exported_count: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        component: ComponentType = ComponentType.PRIVACY_SERVICE
    ) -> bool:
        """Log a privacy request event"""
        try:
            event_id = str(uuid.uuid4())
            
            privacy_event = PrivacyRequestEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                component=component,
                request_type=request_type,
                user_id=user_id,
                request_data=request_data,
                processing_status=processing_status,
                completion_timestamp=completion_timestamp,
                data_deleted_count=data_deleted_count,
                data_exported_count=data_exported_count,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return self.audit_db.store_privacy_request_event(privacy_event)
            
        except Exception as e:
            logger.error(f"Failed to log privacy request event: {e}")
            return False

    def log_system_event(
        self,
        event_type: EventType,
        component: ComponentType,
        description: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> bool:
        """Log a general system event"""
        try:
            event_id = str(uuid.uuid4())
            
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=event_type,
                component=component,
                user_id=user_id,
                session_id=session_id,
                entity_id=None,
                entity_type=None,
                decision_data={"description": description},
                explanation_data=None,
                privacy_impact=PrivacyImpact.NONE,
                ip_address=ip_address,
                user_agent=None,
                success=success,
                error_message=error_message,
                metadata=metadata
            )
            
            return self.audit_db.store_audit_event(audit_event)
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return False

    def log_user_login(
        self,
        user_id: str,
        login_status: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        failure_reason: Optional[str] = None
    ) -> bool:
        """Log user login/logout events"""
        try:
            event_id = str(uuid.uuid4())
            
            metadata = {
                "login_status": login_status,
                "failure_reason": failure_reason
            }
            
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=EventType.USER_LOGIN,
                component=ComponentType.CUSTOMER_REPUTATION,
                user_id=user_id,
                session_id=session_id,
                entity_id=user_id,
                entity_type="user",
                decision_data=metadata,
                explanation_data=None,
                privacy_impact=PrivacyImpact.LOW,
                ip_address=ip_address,
                user_agent=user_agent,
                success=login_status == "success",
                error_message=failure_reason if login_status != "success" else None,
                metadata=metadata
            )
            
            return self.audit_db.store_audit_event(audit_event)
            
        except Exception as e:
            logger.error(f"Failed to log user login event: {e}")
            return False

    def log_config_change(
        self,
        config_item: str,
        old_value: Any,
        new_value: Any,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        component: ComponentType = ComponentType.CUSTOMER_REPUTATION
    ) -> bool:
        """Log configuration changes"""
        try:
            event_id = str(uuid.uuid4())
            
            metadata = {
                "config_item": config_item,
                "old_value": old_value,
                "new_value": new_value
            }
            
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=EventType.CONFIG_CHANGE,
                component=component,
                user_id=user_id,
                session_id=None,
                entity_id=config_item,
                entity_type="config",
                decision_data=metadata,
                explanation_data=None,
                privacy_impact=PrivacyImpact.MEDIUM,
                ip_address=ip_address,
                user_agent=None,
                success=True,
                error_message=None,
                metadata=metadata
            )
            
            return self.audit_db.store_audit_event(audit_event)
            
        except Exception as e:
            logger.error(f"Failed to log config change event: {e}")
            return False

    def log_system_error(
        self,
        error_message: str,
        component: ComponentType,
        error_type: str,
        stack_trace: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None
    ) -> bool:
        """Log system errors"""
        try:
            event_id = str(uuid.uuid4())
            
            metadata = {
                "error_type": error_type,
                "stack_trace": stack_trace
            }
            
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.now(),
                event_type=EventType.SYSTEM_ERROR,
                component=component,
                user_id=user_id,
                session_id=session_id,
                entity_id=entity_id,
                entity_type="error",
                decision_data={"error_message": error_message},
                explanation_data=None,
                privacy_impact=PrivacyImpact.NONE,
                ip_address=None,
                user_agent=None,
                success=False,
                error_message=error_message,
                metadata=metadata
            )
            
            return self.audit_db.store_audit_event(audit_event)
            
        except Exception as e:
            logger.error(f"Failed to log system error event: {e}")
            return False

    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        days: int = 30,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """Get audit trail with filters"""
        try:
            events = self.audit_db.get_audit_events(
                user_id=user_id,
                entity_id=entity_id,
                event_type=event_type,
                component=component,
                start_date=datetime.now() - timedelta(days=days),
                limit=limit
            )
            
            return {
                "success": True,
                "events": events,
                "filters": {
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "event_type": event_type,
                    "component": component,
                    "days": days,
                    "limit": limit
                },
                "total_count": len(events)
            }
            
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return {
                "success": False,
                "error": str(e),
                "events": []
            }

    def export_audit_trail(
        self,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[list] = None
    ) -> Dict[str, Any]:
        """Export audit trail for compliance"""
        try:
            return self.audit_db.export_audit_trail(
                format=format,
                start_date=start_date,
                end_date=end_date,
                event_types=event_types
            )
            
        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get audit statistics"""
        try:
            return self.audit_db.get_audit_statistics(days=days)
            
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def log_decision(**kwargs) -> bool:
    """Convenience function to log decisions"""
    return get_audit_logger().log_decision(**kwargs)


def log_data_access(**kwargs) -> bool:
    """Convenience function to log data access"""
    return get_audit_logger().log_data_access(**kwargs)


def log_model_prediction(**kwargs) -> bool:
    """Convenience function to log model predictions"""
    return get_audit_logger().log_model_prediction(**kwargs)


def log_privacy_request(**kwargs) -> bool:
    """Convenience function to log privacy requests"""
    return get_audit_logger().log_privacy_request(**kwargs)


def log_system_event(**kwargs) -> bool:
    """Convenience function to log system events"""
    return get_audit_logger().log_system_event(**kwargs)