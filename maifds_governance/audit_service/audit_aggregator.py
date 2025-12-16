"""
Audit Aggregator

System-wide audit aggregation and analysis:
- Cross-system event correlation
- Statistical analysis of audit data
- Trend analysis and reporting
- Compliance monitoring
- Performance metrics
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, Counter
import statistics

logger = logging.getLogger(__name__)

@dataclass
class AggregationMetrics:
    """Aggregation metrics"""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_source: Dict[str, int] = field(default_factory=dict)
    events_by_priority: Dict[str, int] = field(default_factory=dict)
    error_rate: float = 0.0
    average_events_per_hour: float = 0.0
    peak_hour: Optional[int] = None
    peak_hour_count: int = 0

@dataclass
class ComplianceMetrics:
    """Compliance monitoring metrics"""
    privacy_violations: int = 0
    security_incidents: int = 0
    access_denials: int = 0
    consent_violations: int = 0
    data_breaches: int = 0
    compliance_score: float = 100.0

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    min_processing_time: float = float('inf')
    throughput_events_per_second: float = 0.0
    queue_depth: int = 0
    system_load: float = 0.0

class AuditAggregator:
    """
    Centralized audit aggregation system
    """
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.aggregation_window = timedelta(hours=24)
        self.last_aggregation = datetime.now()
        
        # Metrics storage
        self.current_metrics = AggregationMetrics()
        self.compliance_metrics = ComplianceMetrics()
        self.performance_metrics = PerformanceMetrics()
        
        # Historical data for trend analysis
        self.historical_metrics: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    def add_event(self, event_data: Dict[str, Any]) -> None:
        """Add event to aggregation"""
        try:
            self.events.append(event_data)
            self._update_real_time_metrics(event_data)
            
            # Check if aggregation is needed
            if datetime.now() - self.last_aggregation > timedelta(minutes=5):
                self.aggregate_events()
                
        except Exception as e:
            logger.error(f"Error adding event to aggregator: {str(e)}")
    
    def _update_real_time_metrics(self, event_data: Dict[str, Any]) -> None:
        """Update real-time metrics with new event"""
        event_type = event_data.get('event_type', 'unknown')
        source = event_data.get('source', 'unknown')
        priority = event_data.get('priority', 1)
        
        # Update counts
        self.current_metrics.total_events += 1
        self.current_metrics.events_by_type[event_type] = self.current_metrics.events_by_type.get(event_type, 0) + 1
        self.current_metrics.events_by_source[source] = self.current_metrics.events_by_source.get(source, 0) + 1
        self.current_metrics.events_by_priority[str(priority)] = self.current_metrics.events_by_priority.get(str(priority), 0) + 1
        
        # Update compliance metrics
        if event_type == 'access_denied':
            self.compliance_metrics.access_denials += 1
        elif event_type == 'privacy_violation':
            self.compliance_metrics.privacy_violations += 1
        elif event_type == 'security_incident':
            self.compliance_metrics.security_incidents += 1
        elif event_type == 'data_breach':
            self.compliance_metrics.data_breaches += 1
    
    def aggregate_events(self) -> Dict[str, Any]:
        """Perform comprehensive event aggregation"""
        try:
            start_time = datetime.now()
            
            # Filter events within aggregation window
            cutoff_time = datetime.now() - self.aggregation_window
            recent_events = [
                event for event in self.events
                if datetime.fromisoformat(event['timestamp']) > cutoff_time
            ]
            
            # Update aggregation metrics
            self._calculate_aggregation_metrics(recent_events)
            self._calculate_compliance_metrics(recent_events)
            self._calculate_performance_metrics(recent_events)
            
            # Store historical snapshot
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'aggregation_metrics': self._metrics_to_dict(self.current_metrics),
                'compliance_metrics': self._metrics_to_dict(self.compliance_metrics),
                'performance_metrics': self._metrics_to_dict(self.performance_metrics),
                'event_count': len(recent_events)
            }
            
            self.historical_metrics.append(snapshot)
            if len(self.historical_metrics) > self.max_history_size:
                self.historical_metrics = self.historical_metrics[-self.max_history_size:]
            
            self.last_aggregation = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Aggregation completed in {processing_time:.2f}s for {len(recent_events)} events")
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            return {}
    
    def _calculate_aggregation_metrics(self, events: List[Dict[str, Any]]) -> None:
        """Calculate basic aggregation metrics"""
        if not events:
            return
        
        # Event type distribution
        type_counts = Counter(event['event_type'] for event in events)
        self.current_metrics.events_by_type = dict(type_counts)
        
        # Source distribution
        source_counts = Counter(event['source'] for event in events)
        self.current_metrics.events_by_source = dict(source_counts)
        
        # Priority distribution
        priority_counts = Counter(str(event['priority']) for event in events)
        self.current_metrics.events_by_priority = dict(priority_counts)
        
        # Hourly distribution
        hour_counts = Counter(
            datetime.fromisoformat(event['timestamp']).hour 
            for event in events
        )
        
        if hour_counts:
            self.current_metrics.peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
            self.current_metrics.peak_hour_count = max(hour_counts.values())
            self.current_metrics.average_events_per_hour = sum(hour_counts.values()) / 24
        
        # Calculate error rate
        error_events = sum(1 for event in events if event.get('event_type') in ['system_error', 'processing_error'])
        self.current_metrics.error_rate = (error_events / len(events)) * 100 if events else 0
    
    def _calculate_compliance_metrics(self, events: List[Dict[str, Any]]) -> None:
        """Calculate compliance metrics"""
        if not events:
            return
        
        # Count compliance-related events
        privacy_violations = sum(1 for event in events if event.get('event_type') == 'privacy_violation')
        security_incidents = sum(1 for event in events if event.get('event_type') == 'security_incident')
        access_denials = sum(1 for event in events if event.get('event_type') == 'access_denied')
        consent_violations = sum(1 for event in events if event.get('event_type') == 'consent_violation')
        data_breaches = sum(1 for event in events if event.get('event_type') == 'data_breach')
        
        self.compliance_metrics.privacy_violations = privacy_violations
        self.compliance_metrics.security_incidents = security_incidents
        self.compliance_metrics.access_denials = access_denials
        self.compliance_metrics.consent_violations = consent_violations
        self.compliance_metrics.data_breaches = data_breaches
        
        # Calculate compliance score (0-100)
        total_compliance_issues = privacy_violations + security_incidents + consent_violations + data_breaches
        total_events = len(events)
        
        if total_events > 0:
            issue_rate = (total_compliance_issues / total_events) * 100
            self.compliance_metrics.compliance_score = max(0, 100 - issue_rate)
    
    def _calculate_performance_metrics(self, events: List[Dict[str, Any]]) -> None:
        """Calculate performance metrics"""
        if not events:
            return
        
        # Extract processing times
        processing_times = []
        for event in events:
            proc_time = event.get('processing_time')
            if proc_time is not None:
                processing_times.append(proc_time)
        
        if processing_times:
            self.performance_metrics.average_processing_time = statistics.mean(processing_times)
            self.performance_metrics.max_processing_time = max(processing_times)
            self.performance_metrics.min_processing_time = min(processing_times)
        
        # Calculate throughput (events per second)
        if len(events) > 1:
            time_span = (
                datetime.fromisoformat(events[-1]['timestamp']) - 
                datetime.fromisoformat(events[0]['timestamp'])
            ).total_seconds()
            
            if time_span > 0:
                self.performance_metrics.throughput_events_per_second = len(events) / time_span
    
    def _metrics_to_dict(self, metrics_obj) -> Dict[str, Any]:
        """Convert metrics object to dictionary"""
        if hasattr(metrics_obj, '__dict__'):
            return {k: v for k, v in metrics_obj.__dict__.items() if not k.startswith('_')}
        return {}
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze trends in audit data"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [
            snapshot for snapshot in self.historical_metrics
            if datetime.fromisoformat(snapshot['timestamp']) > cutoff_time
        ]
        
        if len(recent_snapshots) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Extract trend data
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in recent_snapshots]
        compliance_scores = [s['compliance_metrics']['compliance_score'] for s in recent_snapshots]
        error_rates = [s['aggregation_metrics']['error_rate'] for s in recent_snapshots]
        event_counts = [s['event_count'] for s in recent_snapshots]
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            return (values[-1] - values[0]) / len(values)
        
        return {
            'analysis_period_hours': hours,
            'data_points': len(recent_snapshots),
            'trends': {
                'compliance_score': {
                    'current': compliance_scores[-1] if compliance_scores else 0,
                    'trend': calculate_trend(compliance_scores),
                    'direction': 'improving' if calculate_trend(compliance_scores) > 0 else 'declining'
                },
                'error_rate': {
                    'current': error_rates[-1] if error_rates else 0,
                    'trend': calculate_trend(error_rates),
                    'direction': 'increasing' if calculate_trend(error_rates) > 0 else 'decreasing'
                },
                'event_volume': {
                    'current': event_counts[-1] if event_counts else 0,
                    'trend': calculate_trend(event_counts),
                    'direction': 'increasing' if calculate_trend(event_counts) > 0 else 'decreasing'
                }
            },
            'recommendations': self._generate_recommendations(compliance_scores, error_rates)
        }
    
    def _generate_recommendations(self, compliance_scores: List[float], error_rates: List[float]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if compliance_scores and compliance_scores[-1] < 90:
            recommendations.append("Compliance score below 90% - review privacy and security controls")
        
        if error_rates and error_rates[-1] > 5:
            recommendations.append("High error rate detected - investigate system stability")
        
        # Check for declining trends
        if len(compliance_scores) >= 3:
            recent_trend = (compliance_scores[-1] - compliance_scores[-3]) / 3
            if recent_trend < -5:
                recommendations.append("Declining compliance trend - immediate attention required")
        
        if len(error_rates) >= 3:
            recent_error_trend = (error_rates[-1] - error_rates[-3]) / 3
            if recent_error_trend > 2:
                recommendations.append("Increasing error rate - system performance degrading")
        
        return recommendations
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        # Perform fresh aggregation
        latest_snapshot = self.aggregate_events()
        
        # Get trend analysis
        trend_analysis = self.get_trend_analysis()
        
        # Get top statistics
        top_event_types = sorted(
            self.current_metrics.events_by_type.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_sources = sorted(
            self.current_metrics.events_by_source.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'aggregation_window_hours': self.aggregation_window.total_seconds() / 3600,
            'current_metrics': self._metrics_to_dict(self.current_metrics),
            'compliance_metrics': self._metrics_to_dict(self.compliance_metrics),
            'performance_metrics': self._metrics_to_dict(self.performance_metrics),
            'trend_analysis': trend_analysis,
            'top_statistics': {
                'event_types': dict(top_event_types),
                'sources': dict(top_sources)
            },
            'health_indicators': {
                'overall_health': 'healthy' if self.compliance_metrics.compliance_score > 90 else 'warning',
                'performance_health': 'good' if self.current_metrics.error_rate < 5 else 'poor',
                'compliance_health': 'compliant' if self.compliance_metrics.compliance_score > 85 else 'non_compliant'
            },
            'alerts': self._generate_alerts()
        }
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts based on current metrics"""
        alerts = []
        
        # Compliance alerts
        if self.compliance_metrics.compliance_score < 80:
            alerts.append({
                'type': 'compliance',
                'severity': 'high',
                'message': f"Low compliance score: {self.compliance_metrics.compliance_score:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Error rate alerts
        if self.current_metrics.error_rate > 10:
            alerts.append({
                'type': 'performance',
                'severity': 'high',
                'message': f"High error rate: {self.current_metrics.error_rate:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Security alerts
        if self.compliance_metrics.security_incidents > 5:
            alerts.append({
                'type': 'security',
                'severity': 'critical',
                'message': f"High number of security incidents: {self.compliance_metrics.security_incidents}",
                'timestamp': datetime.now().isoformat()
            })
        
        # Privacy alerts
        if self.compliance_metrics.privacy_violations > 3:
            alerts.append({
                'type': 'privacy',
                'severity': 'high',
                'message': f"Multiple privacy violations: {self.compliance_metrics.privacy_violations}",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def export_report(self, filepath: str, format: str = 'json') -> bool:
        """Export comprehensive report to file"""
        try:
            report = self.get_comprehensive_report()
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'aggregation': self._metrics_to_dict(self.current_metrics),
            'compliance': self._metrics_to_dict(self.compliance_metrics),
            'performance': self._metrics_to_dict(self.performance_metrics),
            'historical_data_points': len(self.historical_metrics),
            'last_aggregation': self.last_aggregation.isoformat(),
            'events_in_window': len([
                e for e in self.events
                if datetime.fromisoformat(e['timestamp']) > (datetime.now() - self.aggregation_window)
            ])
        }
    
    def clear_old_events(self, days: int = 7) -> int:
        """Clear events older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        original_count = len(self.events)
        
        self.events = [
            event for event in self.events
            if datetime.fromisoformat(event['timestamp']) > cutoff_time
        ]
        
        removed_count = original_count - len(self.events)
        logger.info(f"Cleared {removed_count} old events (older than {days} days)")
        
        return removed_count
    
    def set_aggregation_window(self, hours: int) -> None:
        """Set aggregation window in hours"""
        self.aggregation_window = timedelta(hours=hours)
        logger.info(f"Aggregation window set to {hours} hours")