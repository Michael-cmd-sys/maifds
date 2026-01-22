"""
Audit Event Processor

Real-time event processing for audit system:
- Event transformation and enrichment
- Privacy compliance checking
- Anomaly detection in events
- Event routing to appropriate handlers
- Performance monitoring and metrics
"""

from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from collections import defaultdict, deque
import statistics

from .event_bus import Event, EventFilter, EventType, EventPriority, EventBus

logger = logging.getLogger(__name__)

@dataclass
class ProcessingMetrics:
    """Event processing metrics"""
    events_processed: int = 0
    events_failed: int = 0
    average_processing_time: float = 0.0
    processing_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    last_processed: Optional[datetime] = None
    
    def update_processing_time(self, processing_time: float) -> None:
        """Update processing time metrics"""
        self.processing_times.append(processing_time)
        if self.processing_times:
            self.average_processing_time = statistics.mean(self.processing_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'success_rate': (self.events_processed / (self.events_processed + self.events_failed)) if (self.events_processed + self.events_failed) > 0 else 0,
            'average_processing_time': self.average_processing_time,
            'last_processed': self.last_processed.isoformat() if self.last_processed else None,
            'processing_times_count': len(self.processing_times)
        }

class EventProcessor:
    """Base class for event processors"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.metrics = ProcessingMetrics()
    
    async def process(self, event: Event) -> bool:
        """Process event - return True if successful"""
        if not self.enabled:
            return True
        
        start_time = datetime.now()
        
        try:
            result = await self._process_event(event)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics.events_processed += 1
            self.metrics.update_processing_time(processing_time)
            self.metrics.last_processed = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in processor {self.name}: {str(e)}")
            self.metrics.events_failed += 1
            return False
    
    async def _process_event(self, event: Event) -> bool:
        """Override in subclasses"""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            **self.metrics.get_metrics()
        }

class PrivacyComplianceProcessor(EventProcessor):
    """Processor for privacy compliance checking"""
    
    def __init__(self):
        super().__init__("privacy_compliance")
        self.violation_patterns = self._initialize_violation_patterns()
        self.violation_count = defaultdict(int)
    
    def _initialize_violation_patterns(self) -> Dict[str, Callable]:
        """Initialize privacy violation patterns"""
        return {
            'pii_in_logs': self._check_pii_in_logs,
            'unauthorized_access': self._check_unauthorized_access,
            'consent_violation': self._check_consent_violation,
            'data_retention_violation': self._check_data_retention_violation
        }
    
    async def _process_event(self, event: Event) -> bool:
        """Check event for privacy violations"""
        violations = []
        
        for violation_type, check_func in self.violation_patterns.items():
            try:
                if check_func(event):
                    violations.append(violation_type)
                    self.violation_count[violation_type] += 1
            except Exception as e:
                logger.warning(f"Error checking {violation_type}: {str(e)}")
        
        if violations:
            logger.warning(f"Privacy violations detected in event {event.event_id}: {violations}")
            event.metadata['privacy_violations'] = violations
        
        return len(violations) == 0
    
    def _check_pii_in_logs(self, event: Event) -> bool:
        """Check for PII in event data"""
        import re
        
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        event_str = str(event.data) + str(event.metadata)
        
        for pattern in pii_patterns:
            if re.search(pattern, event_str):
                return True
        
        return False
    
    def _check_unauthorized_access(self, event: Event) -> bool:
        """Check for unauthorized access patterns"""
        if event.event_type == EventType.ACCESS_DENIED:
            return True
        
        if event.event_type == EventType.DATA_ACCESS:
            # Check if access was outside business hours
            hour = event.timestamp.hour
            if hour < 9 or hour > 18:
                return True
        
        return False
    
    def _check_consent_violation(self, event: Event) -> bool:
        """Check for consent violations"""
        if event.event_type == EventType.DATA_ACCESS:
            consent_status = event.metadata.get('consent_status')
            if consent_status != 'granted':
                return True
        
        return False
    
    def _check_data_retention_violation(self, event: Event) -> bool:
        """Check for data retention violations"""
        if event.event_type == EventType.DATA_ACCESS:
            data_age = event.metadata.get('data_age_days')
            if data_age and data_age > 2555:  # 7 years
                return True
        
        return False
    
    def get_violation_statistics(self) -> Dict[str, Any]:
        """Get violation statistics"""
        return {
            'total_violations': sum(self.violation_count.values()),
            'violation_types': dict(self.violation_count),
            'most_common_violation': max(self.violation_count.items(), key=lambda x: x[1])[0] if self.violation_count else None
        }

class FileStorageProcessor(EventProcessor):
    """Processor that persists events to a JSONL file"""

    def __init__(self, storage_path: str = "data/audit_events.jsonl"):
        super().__init__("file_storage")
        self.storage_path = storage_path
        self._ensure_dir()

    def _ensure_dir(self):
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    async def _process_event(self, event: Event) -> bool:
        """Write event to file"""
        try:
            import json
            # Serialize properly
            data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type),
                "priority": event.priority.value if hasattr(event.priority, "value") else str(event.priority),
                "source": event.source,
                "user_id": event.user_id,
                "component": event.component,
                "action": event.metadata.get("action", ""),
                "outcome": event.metadata.get("outcome", ""),
                "data": event.data,
                "metadata": event.metadata
            }
            
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write event to file: {e}")
            return False

    def get_recent_logs(self, limit: int = 100) -> List[Dict]:
        """Read recent logs from file (inefficient but works for dashboard)"""
        import json
        logs = []
        try:
            if not os.path.exists(self.storage_path):
                return []
            
            with open(self.storage_path, "r", encoding="utf-8") as f:
                # Read all lines (careful with large files, but okay for prototype)
                lines = f.readlines()
                for line in reversed(lines):
                    if len(logs) >= limit:
                        break
                    try:
                        logs.append(json.loads(line))
                    except:
                        pass
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
        return logs

class AnomalyDetectionProcessor(EventProcessor):
    """Processor for anomaly detection in events"""
    
    def __init__(self):
        super().__init__("anomaly_detection")
        self.baseline_metrics = defaultdict(list)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.anomaly_count = 0
    
    async def _process_event(self, event: Event) -> bool:
        """Detect anomalies in event patterns"""
        anomalies = []
        
        # Check for frequency anomalies
        frequency_anomaly = self._check_frequency_anomaly(event)
        if frequency_anomaly:
            anomalies.append(frequency_anomaly)
        
        # Check for timing anomalies
        timing_anomaly = self._check_timing_anomaly(event)
        if timing_anomaly:
            anomalies.append(timing_anomaly)
        
        # Check for volume anomalies
        volume_anomaly = self._check_volume_anomaly(event)
        if volume_anomaly:
            anomalies.append(volume_anomaly)
        
        if anomalies:
            logger.warning(f"Anomalies detected in event {event.event_id}: {anomalies}")
            event.metadata['anomalies'] = anomalies
            self.anomaly_count += 1
        
        return len(anomalies) == 0
    
    def _check_frequency_anomaly(self, event: Event) -> Optional[str]:
        """Check for frequency anomalies"""
        key = f"{event.event_type.value}_{event.source}"
        current_time = event.timestamp.timestamp()
        
        # Update baseline
        self.baseline_metrics[key].append(current_time)
        
        # Keep only last hour of data
        one_hour_ago = current_time - 3600
        self.baseline_metrics[key] = [
            t for t in self.baseline_metrics[key] if t > one_hour_ago
        ]
        
        # Check frequency
        if len(self.baseline_metrics[key]) > 100:  # More than 100 events per hour
            return f"high_frequency_{key}"
        
        return None
    
    def _check_timing_anomaly(self, event: Event) -> Optional[str]:
        """Check for timing anomalies"""
        hour = event.timestamp.hour
        
        # Check for unusual activity hours
        if event.event_type in [EventType.DATA_ACCESS, EventType.USER_LOGIN]:
            if hour < 6 or hour > 22:
                return f"unusual_hours_{hour}"
        
        return None
    
    def _check_volume_anomaly(self, event: Event) -> Optional[str]:
        """Check for volume anomalies"""
        data_size = len(str(event.data))
        
        # Check for unusually large events
        if data_size > 100000:  # 100KB
            return f"large_event_{data_size}"
        
        return None
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        return {
            'total_anomalies': self.anomaly_count,
            'active_baselines': len(self.baseline_metrics),
            'baseline_metrics': {
                key: len(events) for key, events in self.baseline_metrics.items()
            }
        }

class SecurityEventProcessor(EventProcessor):
    """Processor for security event analysis"""
    
    def __init__(self):
        super().__init__("security_events")
        self.security_patterns = self._initialize_security_patterns()
        self.security_incidents = []
    
    def _initialize_security_patterns(self) -> Dict[str, Callable]:
        """Initialize security pattern checks"""
        return {
            'brute_force': self._check_brute_force,
            'privilege_escalation': self._check_privilege_escalation,
            'data_exfiltration': self._check_data_exfiltration,
            'unusual_behavior': self._check_unusual_behavior
        }
    
    async def _process_event(self, event: Event) -> bool:
        """Analyze event for security patterns"""
        security_issues = []
        
        for pattern_type, check_func in self.security_patterns.items():
            try:
                issue = check_func(event)
                if issue:
                    security_issues.append(issue)
            except Exception as e:
                logger.warning(f"Error checking security pattern {pattern_type}: {str(e)}")
        
        if security_issues:
            logger.warning(f"Security issues detected in event {event.event_id}: {security_issues}")
            event.metadata['security_issues'] = security_issues
            self.security_incidents.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'issues': security_issues
            })
        
        return len(security_issues) == 0
    
    def _check_brute_force(self, event: Event) -> Optional[str]:
        """Check for brute force patterns"""
        if event.event_type == EventType.ACCESS_DENIED:
            user_id = event.user_id
            if user_id:
                # Count recent failed attempts for this user
                recent_failures = [
                    inc for inc in self.security_incidents[-100:]  # Last 100 incidents
                    if inc['timestamp'] > event.timestamp - timedelta(minutes=5)
                    and user_id in str(inc)
                ]
                
                if len(recent_failures) > 5:
                    return f"brute_force_{user_id}"
        
        return None
    
    def _check_privilege_escalation(self, event: Event) -> Optional[str]:
        """Check for privilege escalation"""
        if event.event_type == EventType.USER_LOGIN:
            user_role = event.metadata.get('user_role')
            previous_role = event.metadata.get('previous_role')
            
            if previous_role and user_role and user_role > previous_role:
                return f"privilege_escalation_{previous_role}_to_{user_role}"
        
        return None
    
    def _check_data_exfiltration(self, event: Event) -> Optional[str]:
        """Check for data exfiltration patterns"""
        if event.event_type == EventType.DATA_ACCESS:
            data_size = len(str(event.data))
            access_count = event.metadata.get('access_count', 1)
            
            # Large data access or frequent access
            if data_size > 1000000 or access_count > 100:
                return f"potential_exfiltration_size_{data_size}_count_{access_count}"
        
        return None
    
    def _check_unusual_behavior(self, event: Event) -> Optional[str]:
        """Check for unusual user behavior"""
        if event.event_type == EventType.USER_LOGIN:
            ip_address = event.metadata.get('ip_address')
            user_id = event.user_id
            
            if ip_address and user_id:
                # Check for login from new location
                recent_logins = [
                    inc for inc in self.security_incidents[-50:]
                    if inc['timestamp'] > event.timestamp - timedelta(days=7)
                    and user_id in str(inc)
                ]
                
                known_ips = set()
                for login in recent_logins:
                    for issue in login['issues']:
                        if 'ip_address' in issue:
                            known_ips.add(issue.split('_')[-1])
                
                if ip_address not in known_ips and len(known_ips) > 0:
                    return f"new_location_{user_id}_{ip_address}"
        
        return None
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """Get security analysis statistics"""
        recent_incidents = [
            inc for inc in self.security_incidents
            if inc['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_incidents': len(self.security_incidents),
            'recent_incidents_24h': len(recent_incidents),
            'incident_types': self._analyze_incident_types(),
            'high_risk_users': self._identify_high_risk_users()
        }
    
    def _analyze_incident_types(self) -> Dict[str, int]:
        """Analyze types of security incidents"""
        type_counts = defaultdict(int)
        
        for incident in self.security_incidents:
            for issue in incident['issues']:
                incident_type = issue.split('_')[0]
                type_counts[incident_type] += 1
        
        return dict(type_counts)
    
    def _identify_high_risk_users(self) -> List[str]:
        """Identify users with high security incident rates"""
        user_incidents = defaultdict(int)
        
        for incident in self.security_incidents:
            for issue in incident['issues']:
                if 'user_' in issue:
                    user_id = issue.split('_')[1]
                    user_incidents[user_id] += 1
        
        # Users with more than 5 incidents
        return [user_id for user_id, count in user_incidents.items() if count > 5]

class AuditProcessor:
    """
    Main audit processor that coordinates all event processors
    """
    
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.processors: Dict[str, EventProcessor] = {}
        self.processing_queue = asyncio.Queue()
        self.processing = False
        self.shutdown = False
        
        # Initialize processors
        self._initialize_processors()
    
    def _initialize_processors(self) -> None:
        """Initialize all event processors"""
        self.processors = {
            'privacy_compliance': PrivacyComplianceProcessor(),
            'anomaly_detection': AnomalyDetectionProcessor(),
            'security_events': SecurityEventProcessor(),
            'file_storage': FileStorageProcessor()
        }
        
        # Subscribe to event bus
        for processor_name, processor in self.processors.items():
            self.event_bus.subscribe_filtered(
                EventBus.EventFilter(),
                self._create_processor_callback(processor_name, processor)
            )
    # Backward compatibility for EventBus.EventFilter()
    EventBus.EventFilter = EventFilter

    
    def _create_processor_callback(self, processor_name: str, processor: EventProcessor):
        """Create callback for event processor"""
        async def callback(event: Event):
            await self.processing_queue.put((processor_name, processor, event))
        return callback
    
    async def start_processing(self) -> None:
        """Start audit processing loop"""
        self.processing = True
        logger.info("Audit processor started")
        
        while not self.shutdown:
            try:
                # Get next event to process
                processor_name, processor, event = await asyncio.wait_for(
                    self.processing_queue.get(), timeout=1.0
                )
                
                # Process event
                await processor.process(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in audit processing loop: {str(e)}")
    
    async def stop_processing(self) -> None:
        """Stop audit processing"""
        self.shutdown = True
        self.processing = False
        logger.info("Audit processor stopped")
    
    def get_processor_metrics(self) -> Dict[str, Any]:
        """Get metrics from all processors"""
        return {
            processor_name: processor.get_metrics()
            for processor_name, processor in self.processors.items()
        }
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all processors"""
        stats = {
            'processor_metrics': self.get_processor_metrics(),
            'queue_size': self.processing_queue.qsize(),
            'processing': self.processing
        }
        
        # Add processor-specific statistics
        if 'privacy_compliance' in self.processors:
            privacy_processor = self.processors['privacy_compliance']
            if hasattr(privacy_processor, 'get_violation_statistics'):
                stats['privacy_violations'] = privacy_processor.get_violation_statistics()
        
        if 'anomaly_detection' in self.processors:
            anomaly_processor = self.processors['anomaly_detection']
            if hasattr(anomaly_processor, 'get_anomaly_statistics'):
                stats['anomalies'] = anomaly_processor.get_anomaly_statistics()
        
        if 'security_events' in self.processors:
            security_processor = self.processors['security_events']
            if hasattr(security_processor, 'get_security_statistics'):
                stats['security_incidents'] = security_processor.get_security_statistics()
        
        return stats
    
    def enable_processor(self, processor_name: str) -> bool:
        """Enable a specific processor"""
        if processor_name in self.processors:
            self.processors[processor_name].enabled = True
            logger.info(f"Enabled processor: {processor_name}")
            return True
        return False
    
    def disable_processor(self, processor_name: str) -> bool:
        """Disable a specific processor"""
        if processor_name in self.processors:
            self.processors[processor_name].enabled = False
            logger.info(f"Disabled processor: {processor_name}")
            return True
        return False
    
    def add_custom_processor(self, processor_name: str, processor: EventProcessor) -> None:
        """Add custom event processor"""
        self.processors[processor_name] = processor
        
        # Subscribe to event bus
        self.event_bus.subscribe_filtered(
            self.event_bus.EventFilter(),
            self._create_processor_callback(processor_name, processor)
        )
        
        logger.info(f"Added custom processor: {processor_name}")
    
    def remove_processor(self, processor_name: str) -> bool:
        """Remove a processor"""
        if processor_name in self.processors:
            del self.processors[processor_name]
            logger.info(f"Removed processor: {processor_name}")
            return True
        return False

    def get_audit_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent audit logs from file storage"""
        if 'file_storage' in self.processors:
            return self.processors['file_storage'].get_recent_logs(limit)
        return []