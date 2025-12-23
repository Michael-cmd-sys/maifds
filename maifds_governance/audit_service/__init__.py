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

# ---------------------------------------------------------------------
# FastAPI gateway wrappers (stable function entrypoints)
# ---------------------------------------------------------------------
from typing import Any, Dict, Optional
import inspect
import inspect
import logging
import asyncio
import uuid
from datetime import datetime

# Singletons
_AUDIT_PROCESSOR: Optional[AuditProcessor] = None
_EVENT_BUS: Optional[EventBus] = None
_ALERT_MANAGER: Optional[AlertManager] = None
_CORRELATION_ENGINE: Optional[EventCorrelationEngine] = None
_AGGREGATOR: Optional[AuditAggregator] = None
_MONITOR: Optional[RealTimeMonitor] = None


logger = logging.getLogger(__name__)

def _log_ctor_signature(cls):
    try:
        sig = inspect.signature(cls.__init__)
        logger.error("Constructor signature for %s: %s", cls.__name__, sig)
    except Exception:
        pass


def _construct_with_supported_kwargs(cls, *args, **kwargs):
    """
    Construct cls(*args, **kwargs) but ONLY pass kwargs that exist in cls.__init__ signature.
    This avoids guessing parameter names and avoids unexpected keyword errors.
    """
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    # remove self
    allowed.discard("self")

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return cls(*args, **filtered_kwargs)


def _get_services() -> Dict[str, Any]:
    """
    Build (or reuse) governance singletons in a correct dependency order.
    No guessing. No duplicate kwargs. Matches real constructor signatures.
    """

    global _EVENT_BUS, _ALERT_MANAGER, _CORRELATION_ENGINE, _AGGREGATOR, _AUDIT_PROCESSOR, _MONITOR

    # --- Core primitives ---
    if _EVENT_BUS is None:
        _EVENT_BUS = EventBus()

    if _ALERT_MANAGER is None:
        _ALERT_MANAGER = AlertManager()

    if _CORRELATION_ENGINE is None:
        _CORRELATION_ENGINE = EventCorrelationEngine()

    if _AGGREGATOR is None:
        _AGGREGATOR = AuditAggregator()

    # --- AuditProcessor ---
    # AuditProcessor.__init__(self, event_bus)
    if _AUDIT_PROCESSOR is None:
        _AUDIT_PROCESSOR = AuditProcessor(_EVENT_BUS)

    # --- RealTimeMonitor ---
    # RealTimeMonitor(event_bus, audit_processor, audit_aggregator, alert_manager)
    if _MONITOR is None:
        _MONITOR = RealTimeMonitor(
            event_bus=_EVENT_BUS,
            audit_processor=_AUDIT_PROCESSOR,
            audit_aggregator=_AGGREGATOR,
            alert_manager=_ALERT_MANAGER,
        )

    return {
        "event_bus": _EVENT_BUS,
        "alert_manager": _ALERT_MANAGER,
        "correlation_engine": _CORRELATION_ENGINE,
        "aggregator": _AGGREGATOR,
        "audit_processor": _AUDIT_PROCESSOR,
        "monitor": _MONITOR,
    }



def _call_method_strict(obj: Any, method: str, *args, **kwargs) -> Any:
    """
    Strict: call exactly the method requested; if missing, raise with real available methods.
    """
    fn = getattr(obj, method, None)
    if not callable(fn):
        public = [m for m in dir(obj) if not m.startswith("_")]
        raise AttributeError(f"{obj.__class__.__name__} has no method '{method}'. Available: {public}")
    return fn(*args, **kwargs)


def _safe_event_type(value: str) -> EventType:
    try:
        return EventType(value)
    except Exception:
        return EventType.SYSTEM_ERROR

def _safe_priority(value) -> EventPriority:
    try:
        # allow ints 1..5
        if isinstance(value, int):
            return EventPriority(value)
        # allow strings like "high"
        if isinstance(value, str):
            m = {
                "low": EventPriority.LOW,
                "medium": EventPriority.MEDIUM,
                "high": EventPriority.HIGH,
                "critical": EventPriority.CRITICAL,
                "emergency": EventPriority.EMERGENCY,
            }
            return m.get(value.lower(), EventPriority.MEDIUM)
    except Exception:
        pass
    return EventPriority.MEDIUM


def ingest_event(payload: Dict[str, Any], method: str | None = None) -> Dict[str, Any]:
    """
    Sync wrapper for FastAPI threadpool handlers.
    Safely bridges to async publish() without using create_task().
    """

    services = _get_services()
    bus = services["event_bus"]

    # Keep your custom fields but put them where Event supports them
    meta = dict(payload.get("metadata", {}) or {})
    if "actor_id" in payload:
        meta["actor_id"] = payload.get("actor_id")
    if "action" in payload:
        meta["action"] = payload.get("action")
    if "outcome" in payload:
        meta["outcome"] = payload.get("outcome")

    event = Event(
        event_id=str(uuid.uuid4()),
        event_type=_safe_event_type(payload.get("event_type", "")),
        priority=_safe_priority(payload.get("priority", 2)),
        timestamp=datetime.now(),
        source=payload.get("source", "api"),
        user_id=payload.get("user_id"),
        session_id=payload.get("session_id"),
        component=payload.get("component"),
        data=payload.get("data", {}) or {},
        metadata=meta,
        correlation_id=payload.get("correlation_id"),
        causation_id=payload.get("causation_id"),
    )

    # Thread-safe bridge: run async publish on the server loop
    bus.publish_sync(event)

    return {
        "status": "accepted",
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "priority": event.priority.value,
    }



async def ingest_event_async(payload: Dict[str, Any], method: str | None = None) -> Dict[str, Any]:
    import uuid
    from datetime import datetime
    services = _get_services()
    bus = services["event_bus"]

    event = Event(
        event_id=str(uuid.uuid4()),
        event_type=_safe_event_type(payload.get("event_type", "")),
        priority=_safe_priority(payload.get("priority", 2)),
        timestamp=datetime.now(),
        source=payload.get("source", "api"),
        user_id=payload.get("user_id"),
        session_id=payload.get("session_id"),
        component=payload.get("component"),
        data=payload.get("data", {}),
        metadata=payload.get("metadata", {}),
        correlation_id=payload.get("correlation_id"),
        causation_id=payload.get("causation_id"),
    )

    await bus.publish(event)

    return {"status": "accepted", "event_id": event.event_id}



def get_audit_health() -> Dict[str, Any]:
    """
    Simple sanity endpoint wrapper.
    """
    services = _get_services()
    ap = services["audit_processor"]
    return {
        "status": "ok",
        "audit_processor_class": ap.__class__.__name__,
        "methods": [m for m in dir(ap) if not m.startswith("_")],
    }


def _safe_call(obj, method_name: str, *args, **kwargs):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        return fn(*args, **kwargs)
    return None


def _qsize(obj) -> int | None:
    try:
        return obj.qsize()
    except Exception:
        return None


def get_audit_stats(limit: int = 20) -> Dict[str, Any]:
    """
    Returns observable stats:
    - processor running state
    - queue sizes (if exposed)
    - processor metrics (if available)
    - last N events from EventBus history (if available)
    """
    services = _get_services()
    bus = services["event_bus"]
    ap = services["audit_processor"]
    monitor = services.get("monitor")
    aggregator = services.get("aggregator")
    alert_manager = services.get("alert_manager")

    # --- Event bus observability ---
    # event_bus.py shows these internal fields exist:
    # _event_history (list), _event_queue (asyncio.Queue), _processing (bool)
    bus_processing = getattr(bus, "_processing", None)
    bus_queue = getattr(bus, "_event_queue", None)
    bus_history = getattr(bus, "_event_history", None)

    history_tail = []
    try:
        if isinstance(bus_history, list):
            tail = bus_history[-max(0, int(limit)):] if limit else bus_history
            # Event has to_dict()
            history_tail = [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in tail]
    except Exception as e:
        history_tail = [{"error": f"Failed to read history: {e}"}]

    bus_queue_size = _qsize(bus_queue) if bus_queue is not None else None

    # --- Audit processor observability ---
    ap_processing = getattr(ap, "processing", None)

    # Some processors expose a processing_queue attribute (your /health listed it)
    ap_queue = getattr(ap, "processing_queue", None)
    ap_queue_size = _qsize(ap_queue) if ap_queue is not None else None

    # Metrics methods (exist per your inspect output)
    processor_metrics = _safe_call(ap, "get_processor_metrics")
    comprehensive_stats = _safe_call(ap, "get_comprehensive_statistics")

    # --- Optional: monitor/aggregator/alerts (only if they expose something) ---
    monitor_data = None
    if monitor is not None:
        # try common names without guessing too much
        monitor_data = (
            _safe_call(monitor, "get_dashboard_data")
            or _safe_call(monitor, "get_real_time_metrics")
            or _safe_call(monitor, "get_status")
        )

    aggregator_data = None
    if aggregator is not None:
        aggregator_data = (
            _safe_call(aggregator, "get_aggregation_metrics")
            or _safe_call(aggregator, "get_compliance_metrics")
            or _safe_call(aggregator, "get_performance_metrics")
        )

    alert_data = None
    if alert_manager is not None:
        alert_data = (
            _safe_call(alert_manager, "get_active_alerts")
            or _safe_call(alert_manager, "get_alert_summary")
        )

    return {
        "status": "ok",
        "processor": {
            "class": ap.__class__.__name__,
            "processing": ap_processing,
            "queue_size": ap_queue_size,
        },
        "event_bus": {
            "processing": bus_processing,
            "queue_size": bus_queue_size,
            "history_count": len(bus_history) if isinstance(bus_history, list) else None,
            "last_events": history_tail,
        },
        "metrics": {
            "processor_metrics": processor_metrics,
            "comprehensive_statistics": comprehensive_stats,
        },
        "monitor": monitor_data,
        "aggregator": aggregator_data,
        "alerts": alert_data,
    }
