"""
Alerting System

Real-time alerting for audit and compliance events:
- Multi-severity alert management
- Alert routing and notification
- Alert escalation policies
- Alert aggregation and deduplication
- Integration with external notification systems
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import json
import asyncio
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Alert types"""
    PRIVACY_VIOLATION = "privacy_violation"
    SECURITY_INCIDENT = "security_incident"
    COMPLIANCE_BREACH = "compliance_breach"
    SYSTEM_ERROR = "system_error"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ANOMALY_DETECTED = "anomaly_detected"
    DATA_BREACH = "data_breach"
    ACCESS_DENIED = "access_denied"
    THRESHOLD_EXCEEDED = "threshold_exceeded"

class AlertStatus(Enum):
    """Alert status values"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

@dataclass
class Alert:
    """Alert structure"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    source: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'status': self.status.value,
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'component': self.component,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_by': self.resolved_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalation_level': self.escalation_level,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create alert from dictionary"""
        return cls(
            alert_id=data['alert_id'],
            alert_type=AlertType(data['alert_type']),
            severity=AlertSeverity(data['severity']),
            status=AlertStatus(data['status']),
            title=data['title'],
            description=data['description'],
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            component=data.get('component'),
            correlation_id=data.get('correlation_id'),
            metadata=data.get('metadata', {}),
            acknowledged_by=data.get('acknowledged_by'),
            acknowledged_at=datetime.fromisoformat(data['acknowledged_at']) if data.get('acknowledged_at') else None,
            resolved_by=data.get('resolved_by'),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            escalation_level=data.get('escalation_level', 0),
            tags=data.get('tags', [])
        )

@dataclass
class EscalationPolicy:
    """Alert escalation policy"""
    policy_id: str
    name: str
    alert_types: List[AlertType]
    severity_threshold: AlertSeverity
    escalation_levels: List[Dict[str, Any]]
    max_escalation_level: int = 3
    auto_escalate: bool = True
    escalation_interval_minutes: int = 30

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    name: str
    type: str  # email, sms, slack, webhook, etc.
    config: Dict[str, Any]
    enabled: bool = True
    severity_filter: List[AlertSeverity] = field(default_factory=list)

class AlertManager:
    """
    Comprehensive alert management system
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.alert_history: deque = deque(maxlen=10000)
        
        # Alert deduplication
        self.deduplication_window = timedelta(minutes=5)
        self.recent_alert_signatures: Dict[str, datetime] = {}
        
        # Alert statistics
        self.alert_stats = defaultdict(int)
        self.escalation_queue = asyncio.Queue()
        
        # Initialize default policies
        self._initialize_default_policies()

    def _safe_create_task(self, coro) -> None:
        """Create an async task only if we're inside a running event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # No running loop (called from sync context). Skip background processing.
            # For production you might queue this or run via BackgroundTasks.
            logger.debug("No running event loop; skipping background task.")

    
    def _initialize_default_policies(self) -> None:
        """Initialize default escalation policies"""
        
        # Security incident escalation
        self.escalation_policies['security_incident'] = EscalationPolicy(
            policy_id='security_incident',
            name='Security Incident Escalation',
            alert_types=[AlertType.SECURITY_INCIDENT],
            severity_threshold=AlertSeverity.HIGH,
            escalation_levels=[
                {'level': 1, 'notify': ['security_team'], 'delay_minutes': 0},
                {'level': 2, 'notify': ['security_manager', 'ciso'], 'delay_minutes': 15},
                {'level': 3, 'notify': ['executive_team'], 'delay_minutes': 30}
            ],
            auto_escalate=True,
            escalation_interval_minutes=15
        )
        
        # Privacy violation escalation
        self.escalation_policies['privacy_violation'] = EscalationPolicy(
            policy_id='privacy_violation',
            name='Privacy Violation Escalation',
            alert_types=[AlertType.PRIVACY_VIOLATION],
            severity_threshold=AlertSeverity.MEDIUM,
            escalation_levels=[
                {'level': 1, 'notify': ['privacy_team'], 'delay_minutes': 0},
                {'level': 2, 'notify': ['dpo', 'compliance_officer'], 'delay_minutes': 30},
                {'level': 3, 'notify': ['legal_team'], 'delay_minutes': 60}
            ],
            auto_escalate=True,
            escalation_interval_minutes=30
        )
        
        # Data breach escalation
        self.escalation_policies['data_breach'] = EscalationPolicy(
            policy_id='data_breach',
            name='Data Breach Escalation',
            alert_types=[AlertType.DATA_BREACH],
            severity_threshold=AlertSeverity.CRITICAL,
            escalation_levels=[
                {'level': 1, 'notify': ['incident_response', 'executive_team'], 'delay_minutes': 0},
                {'level': 2, 'notify': ['board', 'regulatory_authorities'], 'delay_minutes': 15}
            ],
            auto_escalate=True,
            escalation_interval_minutes=10
        )
    
    def create_alert(self, 
                   alert_type: AlertType,
                   severity: AlertSeverity,
                   title: str,
                   description: str,
                   source: str,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   component: Optional[str] = None,
                   correlation_id: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   tags: Optional[List[str]] = None) -> Optional[str]:
        """Create new alert"""
        
        # Check for deduplication
        alert_signature = self._generate_alert_signature(
            alert_type, severity, source, component, title
        )
        
        if self._is_duplicate_alert(alert_signature):
            logger.debug(f"Duplicate alert suppressed: {alert_signature}")
            return None
        
        alert_id = str(uuid.uuid4())
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            status=AlertStatus.ACTIVE,
            title=title,
            description=description,
            source=source,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            component=component,
            correlation_id=correlation_id,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Store alert
        self.alerts[alert_id] = alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update deduplication tracking
        self.recent_alert_signatures[alert_signature] = datetime.now()
        
        # Update statistics
        self.alert_stats[f"{alert_type.value}_{severity.value}"] += 1
        
        # Trigger immediate processing
        self._safe_create_task(self._process_new_alert(alert))
        
        logger.warning(f"Alert created: {alert_type.value} - {title}")
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        # Send acknowledgment notification
        self._safe_create_task(self._send_acknowledgment_notification(alert))
        
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.now()
        
        if resolution_notes:
            alert.metadata['resolution_notes'] = resolution_notes
        
        # Remove from active alerts
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        
        # Send resolution notification
        self._safe_create_task(self._send_resolution_notification(alert))
        
        return True
    
    def suppress_alert(self, alert_id: str, reason: str, suppress_until: Optional[datetime] = None) -> bool:
        """Suppress an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        alert.metadata['suppression_reason'] = reason
        alert.metadata['suppress_until'] = suppress_until.isoformat() if suppress_until else None
        
        # Remove from active alerts
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} suppressed: {reason}")
        
        return True
    
    def escalate_alert(self, alert_id: str, escalation_level: int) -> bool:
        """Manually escalate an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.escalation_level = escalation_level
        alert.status = AlertStatus.ESCALATED
        
        logger.warning(f"Alert {alert_id} escalated to level {escalation_level}")
        
        # Send escalation notification
        self._safe_create_task(self._send_escalation_notification(alert))
        
        return True
    
    async def _process_new_alert(self, alert: Alert) -> None:
        """Process new alert"""
        try:
            # Check for immediate escalation
            await self._check_immediate_escalation(alert)
            
            # Send notifications
            await self._send_alert_notifications(alert)
            
            # Schedule escalation if needed
            await self._schedule_escalation(alert)
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {str(e)}")
    
    async def _check_immediate_escalation(self, alert: Alert) -> None:
        """Check if alert requires immediate escalation"""
        for policy in self.escalation_policies.values():
            if (alert.alert_type in policy.alert_types and
                self._severity_meets_threshold(alert.severity, policy.severity_threshold)):
                
                # Immediate escalation to level 1
                if policy.escalation_levels:
                    level_1_config = policy.escalation_levels[0]
                    await self._execute_escalation(alert, level_1_config)
                    alert.escalation_level = 1
    
    def _severity_meets_threshold(self, severity: AlertSeverity, threshold: AlertSeverity) -> bool:
        """Check if severity meets or exceeds threshold"""
        severity_order = {
            AlertSeverity.INFO: 1,
            AlertSeverity.LOW: 2,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.HIGH: 4,
            AlertSeverity.CRITICAL: 5,
            AlertSeverity.EMERGENCY: 6
        }
        
        return severity_order[severity] >= severity_order[threshold]
    
    async def _execute_escalation(self, alert: Alert, escalation_config: Dict[str, Any]) -> None:
        """Execute escalation for alert"""
        notify_targets = escalation_config.get('notify', [])
        
        for target in notify_targets:
            await self._send_escalation_notification(alert, target)
    
    async def _schedule_escalation(self, alert: Alert) -> None:
        """Schedule automatic escalation if needed"""
        # Find applicable policy
        applicable_policy = None
        for policy in self.escalation_policies.values():
            if (alert.alert_type in policy.alert_types and
                self._severity_meets_threshold(alert.severity, policy.severity_threshold)):
                applicable_policy = policy
                break
        
        if not applicable_policy or not applicable_policy.auto_escalate:
            return
        
        # Schedule escalation checks
        for level_config in applicable_policy.escalation_levels[1:]:  # Skip level 1 (immediate)
            delay_minutes = level_config.get('delay_minutes', 30)
            escalation_time = datetime.now() + timedelta(minutes=delay_minutes)
            
            # Add to escalation queue
            await self.escalation_queue.put({
                'alert_id': alert.alert_id,
                'escalation_level': level_config['level'],
                'escalation_time': escalation_time,
                'config': level_config
            })
    
    async def _send_alert_notifications(self, alert: Alert) -> None:
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels.values():
            if not channel.enabled:
                continue
            
            # Check severity filter
            if channel.severity_filter and alert.severity not in channel.severity_filter:
                continue
            
            try:
                await self._send_notification(channel, alert)
            except Exception as e:
                logger.error(f"Error sending notification via {channel.name}: {str(e)}")
    
    async def _send_notification(self, channel: NotificationChannel, alert: Alert) -> None:
        """Send notification through specific channel"""
        if channel.type == 'email':
            await self._send_email_notification(channel, alert)
        elif channel.type == 'slack':
            await self._send_slack_notification(channel, alert)
        elif channel.type == 'webhook':
            await self._send_webhook_notification(channel, alert)
        else:
            logger.warning(f"Unsupported notification channel type: {channel.type}")
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert) -> None:
        """Send email notification"""
        # This would integrate with email service
        logger.info(f"Email notification sent via {channel.name} for alert {alert.alert_id}")
    
    async def _send_slack_notification(self, channel: NotificationChannel, alert: Alert) -> None:
        """Send Slack notification"""
        # This would integrate with Slack API
        logger.info(f"Slack notification sent via {channel.name} for alert {alert.alert_id}")
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert) -> None:
        """Send webhook notification"""
        # This would send HTTP POST to webhook URL
        logger.info(f"Webhook notification sent via {channel.name} for alert {alert.alert_id}")
    
    async def _send_escalation_notification(self, alert: Alert, target: Optional[str] = None) -> None:
        """Send escalation notification"""
        logger.warning(f"Escalation notification for alert {alert.alert_id} to {target or 'default targets'}")
    
    async def _send_acknowledgment_notification(self, alert: Alert) -> None:
        """Send acknowledgment notification"""
        logger.info(f"Acknowledgment notification sent for alert {alert.alert_id}")
    
    async def _send_resolution_notification(self, alert: Alert) -> None:
        """Send resolution notification"""
        logger.info(f"Resolution notification sent for alert {alert.alert_id}")
    
    def _generate_alert_signature(self, 
                               alert_type: AlertType,
                               severity: AlertSeverity,
                               source: str,
                               component: Optional[str],
                               title: str) -> str:
        """Generate signature for deduplication"""
        return f"{alert_type.value}_{severity.value}_{source}_{component or 'none'}_{title}"
    
    def _is_duplicate_alert(self, signature: str) -> bool:
        """Check if alert is duplicate within deduplication window"""
        if signature not in self.recent_alert_signatures:
            return False
        
        last_occurrence = self.recent_alert_signatures[signature]
        return datetime.now() - last_occurrence < self.deduplication_window
    
    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add notification channel"""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.name}")
    
    def remove_notification_channel(self, channel_id: str) -> bool:
        """Remove notification channel"""
        if channel_id in self.notification_channels:
            del self.notification_channels[channel_id]
            logger.info(f"Removed notification channel: {channel_id}")
            return True
        return False
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get specific alert"""
        return self.alerts.get(alert_id)
    
    def get_active_alerts(self, 
                          severity: Optional[AlertSeverity] = None,
                          alert_type: Optional[AlertType] = None,
                          source: Optional[str] = None) -> List[Alert]:
        """Get active alerts with filtering"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.active_alerts)
        
        # Count by status
        status_counts = defaultdict(int)
        for alert in self.alerts.values():
            status_counts[alert.status.value] += 1
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alerts.values():
            severity_counts[alert.severity.value] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in self.alerts.values():
            type_counts[alert.alert_type.value] += 1
        
        # Recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.timestamp > cutoff_time
        ]
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': status_counts.get('resolved', 0),
            'acknowledged_alerts': status_counts.get('acknowledged', 0),
            'suppressed_alerts': status_counts.get('suppressed', 0),
            'recent_alerts_24h': len(recent_alerts),
            'status_distribution': dict(status_counts),
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'notification_channels': len(self.notification_channels),
            'escalation_policies': len(self.escalation_policies),
            'deduplication_window_minutes': self.deduplication_window.total_seconds() / 60
        }
    
    def export_alerts(self, 
                      filepath: str,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None,
                      alert_types: Optional[List[AlertType]] = None) -> int:
        """Export alerts to file"""
        alerts = list(self.alerts.values())
        
        # Apply filters
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        if alert_types:
            alerts = [a for a in alerts if a.alert_type in alert_types]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_alerts': len(alerts),
            'alerts': [alert.to_dict() for alert in alerts]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(alerts)} alerts to {filepath}")
        return len(alerts)
    
    async def start_escalation_monitor(self) -> None:
        """Start escalation monitoring loop"""
        logger.info("Escalation monitor started")
        
        while True:
            try:
                # Check for pending escalations
                escalations = []
                
                # Collect due escalations
                while not self.escalation_queue.empty():
                    try:
                        escalation = self.escalation_queue.get_nowait()
                        if escalation['escalation_time'] <= datetime.now():
                            escalations.append(escalation)
                        else:
                            # Put back if not due yet
                            await self.escalation_queue.put(escalation)
                    except asyncio.QueueEmpty:
                        break
                
                # Execute due escalations
                for escalation in escalations:
                    alert = self.alerts.get(escalation['alert_id'])
                    if alert and alert.status == AlertStatus.ACTIVE:
                        await self._execute_escalation(alert, escalation['config'])
                        alert.escalation_level = escalation['escalation_level']
                        logger.info(f"Alert {alert.alert_id} escalated to level {escalation['escalation_level']}")
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in escalation monitor: {str(e)}")
                await asyncio.sleep(60)
    
    def cleanup_old_alerts(self, days: int = 30) -> int:
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(days=days)
        original_count = len(self.alerts)
        
        # Remove old resolved alerts
        alerts_to_remove = []
        for alert_id, alert in self.alerts.items():
            if alert.status == AlertStatus.RESOLVED:
                if alert.resolved_at and alert.resolved_at < cutoff_time:
                    alerts_to_remove.append(alert_id)
            elif alert.status == AlertStatus.SUPPRESSED:
                # suppressed alerts may not have resolved_at, so use timestamp
                if alert.timestamp < cutoff_time:
                    alerts_to_remove.append(alert_id)
        
        for alert_id in alerts_to_remove:
            del self.alerts[alert_id]
        
        removed_count = len(alerts_to_remove)
        logger.info(f"Cleaned up {removed_count} old alerts (older than {days} days)")
        
        return removed_count