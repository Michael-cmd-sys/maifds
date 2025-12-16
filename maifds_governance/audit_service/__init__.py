"""
Centralized Audit Service

Real-time event processing and system-wide audit aggregation:
- Event bus architecture for unified audit logging
- Real-time event processing and filtering
- System-wide audit aggregation and correlation
- Event-driven privacy and compliance monitoring
- Real-time alerting and anomaly detection
"""

from .event_bus import EventBus, Event, EventType, EventPriority
from .audit_processor import AuditProcessor, EventProcessor
from .audit_aggregator import AuditAggregator
from .alerting import AlertManager, AlertSeverity, AlertType
from .correlation_engine import EventCorrelationEngine
from .real_time_monitor import RealTimeMonitor

__all__ = [
    'EventBus',
    'Event',
    'EventType', 
    'EventPriority',
    'AuditProcessor',
    'EventProcessor',
    'AuditAggregator',
    'AlertManager',
    'AlertSeverity',
    'AlertType',
    'EventCorrelationEngine',
    'RealTimeMonitor'
]