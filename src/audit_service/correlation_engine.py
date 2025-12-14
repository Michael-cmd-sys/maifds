"""
Event Correlation Engine

Advanced event correlation and pattern analysis:
- Temporal correlation of events
- Causal relationship detection
- Pattern recognition and anomaly detection
- Cross-system event correlation
- Attack chain reconstruction
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import statistics

logger = logging.getLogger(__name__)

@dataclass
class CorrelationRule:
    """Event correlation rule"""
    rule_id: str
    name: str
    description: str
    event_types: List[str]
    time_window_minutes: int
    min_events: int
    max_events: Optional[int] = None
    conditions: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"
    active: bool = True

@dataclass
class CorrelatedEvent:
    """Correlated event group"""
    correlation_id: str
    events: List[Dict[str, Any]]
    correlation_type: str
    confidence: float
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"

@dataclass
class AttackPattern:
    """Attack pattern definition"""
    pattern_id: str
    name: str
    description: str
    stages: List[Dict[str, Any]]
    time_window_minutes: int
    severity: str = "high"

class EventCorrelationEngine:
    """
    Advanced event correlation engine
    """
    
    def __init__(self):
        self.correlation_rules: Dict[str, CorrelationRule] = {}
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.event_buffer: deque = deque(maxlen=10000)
        self.correlated_events: Dict[str, CorrelatedEvent] = {}
        self.correlation_history: List[CorrelatedEvent] = []
        
        # Correlation statistics
        self.correlation_stats = defaultdict(int)
        self.pattern_matches = defaultdict(int)
        
        # Initialize default correlation rules
        self._initialize_default_rules()
        self._initialize_attack_patterns()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default correlation rules"""
        
        # Failed login attempts correlation
        self.correlation_rules['failed_logins'] = CorrelationRule(
            rule_id='failed_logins',
            name='Multiple Failed Login Attempts',
            description='Detects multiple failed login attempts',
            event_types=['access_denied'],
            time_window_minutes=15,
            min_events=5,
            severity='high'
        )
        
        # Privilege escalation correlation
        self.correlation_rules['privilege_escalation'] = CorrelationRule(
            rule_id='privilege_escalation',
            name='Privilege Escalation Pattern',
            description='Detects privilege escalation attempts',
            event_types=['user_login', 'config_change'],
            time_window_minutes=30,
            min_events=2,
            conditions={'role_change': True}
        )
        
        # Data access correlation
        self.correlation_rules['unusual_data_access'] = CorrelationRule(
            rule_id='unusual_data_access',
            name='Unusual Data Access Pattern',
            description='Detects unusual data access patterns',
            event_types=['data_access'],
            time_window_minutes=60,
            min_events=10,
            conditions={'volume_threshold': 100}
        )
        
        # Security incident correlation
        self.correlation_rules['security_incident_chain'] = CorrelationRule(
            rule_id='security_incident_chain',
            name='Security Incident Chain',
            description='Detects chains of security incidents',
            event_types=['security_incident', 'anomaly_detected'],
            time_window_minutes=120,
            min_events=3,
            severity='critical'
        )
    
    def _initialize_attack_patterns(self) -> None:
        """Initialize attack pattern definitions"""
        
        # Brute force attack pattern
        self.attack_patterns['brute_force'] = AttackPattern(
            pattern_id='brute_force',
            name='Brute Force Attack',
            description='Multiple failed login attempts followed by successful login',
            stages=[
                {'event_type': 'access_denied', 'min_count': 5, 'time_window': 10},
                {'event_type': 'user_login', 'min_count': 1, 'time_window': 5}
            ],
            time_window_minutes=30,
            severity='high'
        )
        
        # Data exfiltration pattern
        self.attack_patterns['data_exfiltration'] = AttackPattern(
            pattern_id='data_exfiltration',
            name='Data Exfiltration',
            description='Large data access followed by unusual export activity',
            stages=[
                {'event_type': 'data_access', 'min_count': 50, 'time_window': 60},
                {'event_type': 'system_error', 'pattern': 'export*', 'min_count': 1, 'time_window': 30}
            ],
            time_window_minutes=120,
            severity='critical'
        )
        
        # Insider threat pattern
        self.attack_patterns['insider_threat'] = AttackPattern(
            pattern_id='insider_threat',
            name='Insider Threat Pattern',
            description='Unusual access patterns combined with privilege escalation',
            stages=[
                {'event_type': 'data_access', 'unusual_time': True, 'min_count': 1},
                {'event_type': 'user_login', 'unusual_location': True, 'min_count': 1},
                {'event_type': 'config_change', 'privilege_change': True, 'min_count': 1}
            ],
            time_window_minutes=240,
            severity='critical'
        )
    
    def add_event(self, event_data: Dict[str, Any]) -> None:
        """Add event for correlation analysis"""
        try:
            # Add to buffer
            self.event_buffer.append(event_data)
            
            # Perform correlation analysis
            correlations = self._correlate_event(event_data)
            
            # Store correlations
            for correlation in correlations:
                self.correlated_events[correlation.correlation_id] = correlation
                self.correlation_history.append(correlation)
                self.correlation_stats[correlation.correlation_type] += 1
            
            # Check for attack patterns
            pattern_matches = self._detect_attack_patterns(event_data)
            for pattern_id, match_info in pattern_matches.items():
                self.pattern_matches[pattern_id] += 1
                logger.warning(f"Attack pattern detected: {pattern_id} - {match_info}")
                
        except Exception as e:
            logger.error(f"Error in event correlation: {str(e)}")
    
    def _correlate_event(self, event_data: Dict[str, Any]) -> List[CorrelatedEvent]:
        """Correlate event with existing events"""
        correlations = []
        
        for rule in self.correlation_rules.values():
            if not rule.active:
                continue
            
            # Check if event type matches rule
            if event_data.get('event_type') not in rule.event_types:
                continue
            
            # Find related events within time window
            related_events = self._find_related_events(event_data, rule)
            
            # Check if correlation conditions are met
            if self._meets_correlation_conditions(related_events, rule):
                correlation = self._create_correlation(related_events, rule)
                correlations.append(correlation)
        
        return correlations
    
    def _find_related_events(self, event_data: Dict[str, Any], rule: CorrelationRule) -> List[Dict[str, Any]]:
        """Find events related to the given event"""
        event_time = datetime.fromisoformat(event_data['timestamp'])
        time_window = timedelta(minutes=rule.time_window_minutes)
        
        related_events = []
        
        # Search in event buffer
        for buffered_event in self.event_buffer:
            buffered_time = datetime.fromisoformat(buffered_event['timestamp'])
            
            # Check time window
            if abs((event_time - buffered_time).total_seconds()) > time_window.total_seconds():
                continue
            
            # Check event type
            if buffered_event.get('event_type') in rule.event_types:
                related_events.append(buffered_event)
        
        # Include current event
        related_events.append(event_data)
        
        return related_events
    
    def _meets_correlation_conditions(self, events: List[Dict[str, Any]], rule: CorrelationRule) -> bool:
        """Check if events meet correlation conditions"""
        event_count = len(events)
        
        # Check minimum event count
        if event_count < rule.min_events:
            return False
        
        # Check maximum event count
        if rule.max_events and event_count > rule.max_events:
            return False
        
        # Check additional conditions
        for condition_key, condition_value in rule.conditions.items():
            if not self._evaluate_condition(events, condition_key, condition_value):
                return False
        
        return True
    
    def _evaluate_condition(self, events: List[Dict[str, Any]], condition_key: str, condition_value: Any) -> bool:
        """Evaluate specific correlation condition"""
        if condition_key == 'role_change':
            # Check for role change events
            role_changes = [
                event for event in events
                if event.get('metadata', {}).get('role_change')
            ]
            return len(role_changes) >= condition_value
        
        elif condition_key == 'volume_threshold':
            # Check data access volume
            total_volume = sum(
                event.get('metadata', {}).get('data_size', 0)
                for event in events
            )
            return total_volume >= condition_value
        
        elif condition_key == 'same_user':
            # Check if all events are from same user
            users = set(event.get('user_id') for event in events if event.get('user_id'))
            return len(users) <= condition_value
        
        elif condition_key == 'same_source':
            # Check if all events are from same source
            sources = set(event.get('source') for event in events)
            return len(sources) <= condition_value
        
        return True
    
    def _create_correlation(self, events: List[Dict[str, Any]], rule: CorrelationRule) -> CorrelatedEvent:
        """Create correlation from related events"""
        # Sort events by timestamp
        events.sort(key=lambda e: datetime.fromisoformat(e['timestamp']))
        
        start_time = datetime.fromisoformat(events[0]['timestamp'])
        end_time = datetime.fromisoformat(events[-1]['timestamp'])
        
        # Calculate confidence based on event count and rule match
        confidence = min(1.0, len(events) / (rule.min_events * 2))
        
        correlation_id = f"corr_{datetime.now().timestamp()}_{rule.rule_id}"
        
        return CorrelatedEvent(
            correlation_id=correlation_id,
            events=events,
            correlation_type=rule.name,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            metadata={
                'rule_id': rule.rule_id,
                'event_count': len(events),
                'time_span_minutes': (end_time - start_time).total_seconds() / 60
            },
            severity=rule.severity
        )
    
    def _detect_attack_patterns(self, event_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Detect attack patterns in event stream"""
        pattern_matches = {}
        
        for pattern in self.attack_patterns.values():
            if self._matches_attack_pattern(event_data, pattern):
                match_info = {
                    'pattern_id': pattern.pattern_id,
                    'pattern_name': pattern.name,
                    'detected_at': datetime.now().isoformat(),
                    'triggering_event': event_data.get('event_id'),
                    'severity': pattern.severity
                }
                pattern_matches[pattern.pattern_id] = match_info
        
        return pattern_matches
    
    def _matches_attack_pattern(self, event_data: Dict[str, Any], pattern: AttackPattern) -> bool:
        """Check if event matches attack pattern"""
        event_time = datetime.fromisoformat(event_data['timestamp'])
        time_window = timedelta(minutes=pattern.time_window_minutes)
        
        # Check each stage of the attack pattern
        for stage in pattern.stages:
            stage_events = []
            required_event_type = stage['event_type']
            min_count = stage.get('min_count', 1)
            
            # Find events matching this stage
            for buffered_event in self.event_buffer:
                buffered_time = datetime.fromisoformat(buffered_event['timestamp'])
                
                # Check time window
                if abs((event_time - buffered_time).total_seconds()) > time_window.total_seconds():
                    continue
                
                # Check event type
                if buffered_event.get('event_type') == required_event_type:
                    # Check additional stage conditions
                    if self._meets_stage_conditions(buffered_event, stage):
                        stage_events.append(buffered_event)
            
            # Check if stage requirements are met
            if len(stage_events) < min_count:
                return False
        
        return True
    
    def _meets_stage_conditions(self, event: Dict[str, Any], stage: Dict[str, Any]) -> bool:
        """Check if event meets stage conditions"""
        if 'unusual_time' in stage:
            hour = datetime.fromisoformat(event['timestamp']).hour
            if stage['unusual_time'] and 9 <= hour <= 17:  # Business hours
                return False
        
        if 'unusual_location' in stage:
            location = event.get('metadata', {}).get('location')
            if stage['unusual_location'] and location in ['US', 'EU', 'office']:  # Normal locations
                return False
        
        if 'privilege_change' in stage:
            role_change = event.get('metadata', {}).get('role_change')
            if stage['privilege_change'] and not role_change:
                return False
        
        if 'pattern' in stage:
            # Check if event description matches pattern
            description = event.get('description', '')
            import re
            if not re.search(stage['pattern'], description, re.IGNORECASE):
                return False
        
        return True
    
    def get_correlations(self, 
                        correlation_type: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        severity: Optional[str] = None) -> List[CorrelatedEvent]:
        """Get correlated events with filtering"""
        correlations = list(self.correlated_events.values())
        
        if correlation_type:
            correlations = [c for c in correlations if c.correlation_type == correlation_type]
        if start_time:
            correlations = [c for c in correlations if c.start_time >= start_time]
        if end_time:
            correlations = [c for c in correlations if c.end_time <= end_time]
        if severity:
            correlations = [c for c in correlations if c.severity == severity]
        
        # Sort by start time (newest first)
        correlations.sort(key=lambda c: c.start_time, reverse=True)
        
        return correlations
    
    def get_attack_pattern_matches(self, 
                                 pattern_id: Optional[str] = None,
                                 hours: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent attack pattern matches"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_matches = defaultdict(list)
        
        # This would typically query a database of pattern matches
        # For now, return current pattern match statistics
        for pattern_id_key, count in self.pattern_matches.items():
            if pattern_id is None or pattern_id_key == pattern_id:
                recent_matches[pattern_id_key] = [
                    {
                        'pattern_id': pattern_id_key,
                        'match_count': count,
                        'last_match': datetime.now().isoformat()
                    }
                ]
        
        return dict(recent_matches)
    
    def get_correlation_statistics(self) -> Dict[str, Any]:
        """Get correlation engine statistics"""
        total_correlations = len(self.correlation_history)
        
        # Correlation type distribution
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for correlation in self.correlation_history:
            type_counts[correlation.correlation_type] += 1
            severity_counts[correlation.severity] += 1
        
        # Average confidence by type
        confidence_by_type = defaultdict(list)
        for correlation in self.correlation_history:
            confidence_by_type[correlation.correlation_type].append(correlation.confidence)
        
        avg_confidence = {}
        for corr_type, confidences in confidence_by_type.items():
            avg_confidence[corr_type] = statistics.mean(confidences) if confidences else 0
        
        return {
            'total_correlations': total_correlations,
            'active_correlations': len(self.correlated_events),
            'correlation_rules': len(self.correlation_rules),
            'attack_patterns': len(self.attack_patterns),
            'event_buffer_size': len(self.event_buffer),
            'correlation_types': dict(type_counts),
            'severity_distribution': dict(severity_counts),
            'average_confidence_by_type': avg_confidence,
            'pattern_matches': dict(self.pattern_matches),
            'most_common_correlation': max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
            'most_active_pattern': max(self.pattern_matches.items(), key=lambda x: x[1])[0] if self.pattern_matches else None
        }
    
    def add_correlation_rule(self, rule: CorrelationRule) -> None:
        """Add custom correlation rule"""
        self.correlation_rules[rule.rule_id] = rule
        logger.info(f"Added correlation rule: {rule.name}")
    
    def remove_correlation_rule(self, rule_id: str) -> bool:
        """Remove correlation rule"""
        if rule_id in self.correlation_rules:
            del self.correlation_rules[rule_id]
            logger.info(f"Removed correlation rule: {rule_id}")
            return True
        return False
    
    def add_attack_pattern(self, pattern: AttackPattern) -> None:
        """Add custom attack pattern"""
        self.attack_patterns[pattern.pattern_id] = pattern
        logger.info(f"Added attack pattern: {pattern.name}")
    
    def remove_attack_pattern(self, pattern_id: str) -> bool:
        """Remove attack pattern"""
        if pattern_id in self.attack_patterns:
            del self.attack_patterns[pattern_id]
            logger.info(f"Removed attack pattern: {pattern_id}")
            return True
        return False
    
    def export_correlations(self, 
                           filepath: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> int:
        """Export correlations to file"""
        correlations = self.get_correlations(start_time=start_time, end_time=end_time)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_correlations': len(correlations),
            'correlations': [
                {
                    'correlation_id': c.correlation_id,
                    'correlation_type': c.correlation_type,
                    'confidence': c.confidence,
                    'severity': c.severity,
                    'start_time': c.start_time.isoformat(),
                    'end_time': c.end_time.isoformat(),
                    'event_count': len(c.events),
                    'metadata': c.metadata,
                    'events': c.events
                }
                for c in correlations
            ]
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(correlations)} correlations to {filepath}")
        return len(correlations)
    
    def clear_old_correlations(self, days: int = 7) -> int:
        """Clear old correlations"""
        cutoff_time = datetime.now() - timedelta(days=days)
        original_count = len(self.correlation_history)
        
        self.correlation_history = [
            c for c in self.correlation_history
            if c.start_time > cutoff_time
        ]
        
        removed_count = original_count - len(self.correlation_history)
        logger.info(f"Cleared {removed_count} old correlations (older than {days} days)")
        
        return removed_count
    
    def get_correlation_timeline(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get timeline of correlations"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_correlations = [
            c for c in self.correlation_history
            if c.start_time > cutoff_time
        ]
        
        # Group by hour
        timeline = defaultdict(list)
        for correlation in recent_correlations:
            hour = correlation.start_time.hour
            timeline[hour].append(correlation)
        
        # Create timeline data
        timeline_data = []
        for hour in range(24):
            hour_correlations = timeline.get(hour, [])
            timeline_data.append({
                'hour': hour,
                'correlation_count': len(hour_correlations),
                'severity_breakdown': {
                    severity: len([c for c in hour_correlations if c.severity == severity])
                    for severity in set(c.severity for c in hour_correlations)
                },
                'type_breakdown': {
                    corr_type: len([c for c in hour_correlations if c.correlation_type == corr_type])
                    for corr_type in set(c.correlation_type for c in hour_correlations)
                }
            })
        
        return timeline_data