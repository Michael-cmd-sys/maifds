"""
Real-time Monitor

Real-time monitoring and dashboard for audit system:
- Live event streaming
- Real-time metrics dashboard
- Performance monitoring
- System health monitoring
- Alert visualization
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import logging
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class SystemHealth:
    """System health status"""
    overall_status: str = "healthy"
    event_bus_status: str = "healthy"
    processor_status: str = "healthy"
    aggregator_status: str = "healthy"
    alerting_status: str = "healthy"
    correlation_status: str = "healthy"
    last_check: datetime = field(default_factory=datetime.now)
    issues: List[str] = field(default_factory=list)

@dataclass
class RealTimeMetrics:
    """Real-time system metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    events_per_second: float = 0.0
    processing_latency_ms: float = 0.0
    queue_depth: int = 0
    active_alerts: int = 0
    system_cpu_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mbps: float = 0.0

@dataclass
class DashboardData:
    """Dashboard data structure"""
    metrics: RealTimeMetrics = field(default_factory=RealTimeMetrics)
    health: SystemHealth = field(default_factory=SystemHealth)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    top_sources: List[Dict[str, Any]] = field(default_factory=list)
    event_types: Dict[str, int] = field(default_factory=dict)
    compliance_score: float = 100.0

class RealTimeMonitor:
    """
    Real-time monitoring system
    """
    
    def __init__(self, event_bus, audit_processor, audit_aggregator, alert_manager):
        self.event_bus = event_bus
        self.audit_processor = audit_processor
        self.audit_aggregator = audit_aggregator
        self.alert_manager = alert_manager
        
        # Monitoring data
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = RealTimeMetrics()
        self.system_health = SystemHealth()
        self.dashboard_data = DashboardData()
        
        # Monitoring configuration
        self.monitoring_interval = 5  # seconds
        self.health_check_interval = 30  # seconds
        self.metrics_window = 300  # 5 minutes for rolling metrics
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[Any] = set()
        
        # Monitoring state
        self.monitoring_active = False
        self.shutdown_requested = False
        
        # Callbacks for custom monitoring
        self.metrics_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        logger.info("Starting real-time monitoring")
        self.monitoring_active = True
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._dashboard_update_loop()),
            asyncio.create_task(self._alert_monitoring_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Real-time monitoring stopped")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        logger.info("Stopping real-time monitoring")
        self.shutdown_requested = True
        self.monitoring_active = False
    
    async def _metrics_collection_loop(self) -> None:
        """Collect system metrics periodically"""
        while not self.shutdown_requested:
            try:
                # Collect current metrics
                metrics = await self._collect_system_metrics()
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Trigger metrics callbacks
                for callback in self.metrics_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(metrics)
                        else:
                            callback(metrics)
                    except Exception as e:
                        logger.error(f"Error in metrics callback: {str(e)}")
                
                # Broadcast to WebSocket connections
                await self._broadcast_metrics_update(metrics)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {str(e)}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _health_check_loop(self) -> None:
        """Perform system health checks"""
        while not self.shutdown_requested:
            try:
                health = await self._perform_health_checks()
                self.system_health = health
                
                # Log health changes
                if health.overall_status != "healthy":
                    logger.warning(f"System health degraded: {health.overall_status} - Issues: {health.issues}")
                
            except Exception as e:
                logger.error(f"Error in health check: {str(e)}")
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _dashboard_update_loop(self) -> None:
        """Update dashboard data periodically"""
        while not self.shutdown_requested:
            try:
                # Get recent events
                recent_events = self.event_bus.get_events(limit=50)
                
                # Get active alerts
                active_alerts = self.alert_manager.get_active_alerts()
                
                # Get top sources
                event_stats = self.event_bus.get_event_statistics()
                top_sources = sorted(
                    event_stats.get('sources', {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                # Update dashboard data
                self.dashboard_data = DashboardData(
                    metrics=self.current_metrics,
                    health=self.system_health,
                    recent_events=[event.to_dict() if hasattr(event, 'to_dict') else event for event in recent_events[:20]],
                    active_alerts=[alert.to_dict() if hasattr(alert, 'to_dict') else alert for alert in active_alerts[:10]],
                    top_sources=[{'source': k, 'count': v} for k, v in top_sources],
                    event_types=event_stats.get('event_types', {}),
                    compliance_score=self._calculate_compliance_score()
                )
                
                # Broadcast dashboard update
                await self._broadcast_dashboard_update()
                
            except Exception as e:
                logger.error(f"Error in dashboard update: {str(e)}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _alert_monitoring_loop(self) -> None:
        """Monitor for new alerts"""
        while not self.shutdown_requested:
            try:
                # Get recent alerts
                active_alerts = self.alert_manager.get_active_alerts()
                
                # Check for new high-severity alerts
                high_severity_alerts = [
                    alert for alert in active_alerts
                    if alert.severity.value in ['high', 'critical', 'emergency']
                ]
                
                # Trigger alert callbacks for new alerts
                current_alert_ids = set(alert.alert_id for alert in active_alerts)
                previous_alert_ids = set(getattr(self, '_last_alert_ids', set()))
                
                new_alert_ids = current_alert_ids - previous_alert_ids
                if new_alert_ids:
                    new_alerts = [alert for alert in active_alerts if alert.alert_id in new_alert_ids]
                    
                    for callback in self.alert_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(new_alerts)
                            else:
                                callback(new_alerts)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {str(e)}")
                
                self._last_alert_ids = current_alert_ids
                
            except Exception as e:
                logger.error(f"Error in alert monitoring: {str(e)}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> RealTimeMetrics:
        """Collect current system metrics"""
        try:
            # Event processing metrics
            event_stats = self.event_bus.get_event_statistics()
            processor_stats = self.audit_processor.get_processor_metrics()
            
            # Calculate events per second
            events_per_second = 0.0
            if len(self.metrics_history) > 1:
                recent_metrics = list(self.metrics_history)[-12:]  # Last minute
                if len(recent_metrics) >= 2:
                    event_count_diff = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
                    time_diff = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
                    if time_diff > 0:
                        events_per_second = event_count_diff / time_diff
            
            # Calculate processing latency
            processing_latency = 0.0
            if processor_stats:
                latencies = []
                for processor_name, stats in processor_stats.items():
                    if 'average_processing_time' in stats:
                        latencies.append(stats['average_processing_time'] * 1000)  # Convert to ms
                
                if latencies:
                    processing_latency = statistics.mean(latencies)
            
            # Get queue depth
            queue_depth = event_stats.get('queue_size', 0)
            
            # Get active alerts count
            active_alerts = len(self.alert_manager.get_active_alerts())
            
            # System resource metrics (placeholder - would use psutil in real implementation)
            system_cpu = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            disk_usage = self._get_disk_usage()
            network_io = self._get_network_io()
            
            return RealTimeMetrics(
                timestamp=datetime.now(),
                events_per_second=events_per_second,
                processing_latency_ms=processing_latency,
                queue_depth=queue_depth,
                active_alerts=active_alerts,
                system_cpu_percent=system_cpu,
                memory_usage_percent=memory_usage,
                disk_usage_percent=disk_usage,
                network_io_mbps=network_io
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return RealTimeMetrics()
    
    async def _perform_health_checks(self) -> SystemHealth:
        """Perform comprehensive health checks"""
        issues = []
        
        # Check event bus health
        event_bus_stats = self.event_bus.get_event_statistics()
        event_bus_healthy = (
            event_bus_stats.get('processing', True) and
            event_bus_stats.get('queue_size', 0) < 10000
        )
        
        if not event_bus_healthy:
            issues.append("Event bus overloaded or not processing")
        
        # Check processor health
        processor_stats = self.audit_processor.get_processor_metrics()
        processor_healthy = all(
            stats.get('enabled', True) and
            stats.get('success_rate', 1.0) > 0.9
            for stats in processor_stats.values()
        )
        
        if not processor_healthy:
            issues.append("Processors have low success rate or are disabled")
        
        # Check aggregator health
        aggregator_metrics = self.audit_aggregator.get_metrics_summary()
        aggregator_healthy = (
            aggregator_metrics.get('compliance', {}).get('compliance_score', 100) > 80
        )
        
        if not aggregator_healthy:
            issues.append("Aggregator shows low compliance score")
        
        # Check alerting health
        alert_stats = self.alert_manager.get_alert_statistics()
        alerting_healthy = (
            alert_stats.get('active_alerts', 0) < 100  # Reasonable alert limit
        )
        
        if not alerting_healthy:
            issues.append("Alerting system has too many active alerts")
        
        # Determine overall status
        if not issues:
            overall_status = "healthy"
        elif len(issues) <= 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        return SystemHealth(
            overall_status=overall_status,
            event_bus_status="healthy" if event_bus_healthy else "unhealthy",
            processor_status="healthy" if processor_healthy else "unhealthy",
            aggregator_status="healthy" if aggregator_healthy else "unhealthy",
            alerting_status="healthy" if alerting_healthy else "unhealthy",
            correlation_status="healthy",  # Would check correlation engine
            last_check=datetime.now(),
            issues=issues
        )
    
    def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score"""
        try:
            # Get compliance metrics from aggregator
            metrics = self.audit_aggregator.get_metrics_summary()
            compliance_metrics = metrics.get('compliance', {})
            
            base_score = compliance_metrics.get('compliance_score', 100)
            
            # Adjust based on active alerts
            active_alerts = len(self.alert_manager.get_active_alerts())
            alert_penalty = min(20, active_alerts * 2)  # 2 points per alert, max 20
            
            # Adjust based on system health
            health_penalty = 0
            if self.system_health.overall_status == "degraded":
                health_penalty = 10
            elif self.system_health.overall_status == "unhealthy":
                health_penalty = 25
            
            final_score = max(0, base_score - alert_penalty - health_penalty)
            
            return round(final_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {str(e)}")
            return 100.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        # Placeholder implementation
        import random
        return random.uniform(10, 80)
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        # Placeholder implementation
        import random
        return random.uniform(30, 90)
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        # Placeholder implementation
        import random
        return random.uniform(20, 85)
    
    def _get_network_io(self) -> float:
        """Get network I/O in Mbps"""
        # Placeholder implementation
        import random
        return random.uniform(1, 100)
    
    async def _broadcast_metrics_update(self, metrics: RealTimeMetrics) -> None:
        """Broadcast metrics update to WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = {
            'type': 'metrics_update',
            'data': {
                'timestamp': metrics.timestamp.isoformat(),
                'events_per_second': metrics.events_per_second,
                'processing_latency_ms': metrics.processing_latency_ms,
                'queue_depth': metrics.queue_depth,
                'active_alerts': metrics.active_alerts,
                'system_cpu_percent': metrics.system_cpu_percent,
                'memory_usage_percent': metrics.memory_usage_percent,
                'disk_usage_percent': metrics.disk_usage_percent,
                'network_io_mbps': metrics.network_io_mbps
            }
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_dashboard_update(self) -> None:
        """Broadcast dashboard update to WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = {
            'type': 'dashboard_update',
            'data': {
                'metrics': {
                    'timestamp': self.dashboard_data.metrics.timestamp.isoformat(),
                    'events_per_second': self.dashboard_data.metrics.events_per_second,
                    'processing_latency_ms': self.dashboard_data.metrics.processing_latency_ms,
                    'queue_depth': self.dashboard_data.metrics.queue_depth,
                    'active_alerts': self.dashboard_data.metrics.active_alerts
                },
                'health': {
                    'overall_status': self.dashboard_data.health.overall_status,
                    'last_check': self.dashboard_data.health.last_check.isoformat(),
                    'issues': self.dashboard_data.health.issues
                },
                'recent_events': self.dashboard_data.recent_events,
                'active_alerts': self.dashboard_data.active_alerts,
                'top_sources': self.dashboard_data.top_sources,
                'event_types': self.dashboard_data.event_types,
                'compliance_score': self.dashboard_data.compliance_score
            }
        }
        
        await self._broadcast_to_websockets(message)
    
    async def _broadcast_to_websockets(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all WebSocket connections"""
        message_str = json.dumps(message)
        
        # Send to each connection
        for connection in self.websocket_connections.copy():
            try:
                await connection.send(message_str)
            except Exception as e:
                logger.warning(f"Error sending to WebSocket: {str(e)}")
                # Remove failed connection
                self.websocket_connections.discard(connection)
    
    def add_websocket_connection(self, connection: Any) -> None:
        """Add WebSocket connection"""
        self.websocket_connections.add(connection)
        logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
    
    def remove_websocket_connection(self, connection: Any) -> None:
        """Remove WebSocket connection"""
        self.websocket_connections.discard(connection)
        logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
    
    def add_metrics_callback(self, callback: Callable) -> None:
        """Add custom metrics callback"""
        self.metrics_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add custom alert callback"""
        self.alert_callbacks.append(callback)
    
    def get_current_dashboard_data(self) -> DashboardData:
        """Get current dashboard data"""
        return self.dashboard_data
    
    def get_metrics_history(self, minutes: int = 60) -> List[RealTimeMetrics]:
        """Get metrics history"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        return self.system_health
    
    def export_monitoring_data(self, filepath: str, hours: int = 24) -> bool:
        """Export monitoring data to file"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter data
            recent_metrics = [
                metrics for metrics in self.metrics_history
                if metrics.timestamp > cutoff_time
            ]
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'export_period_hours': hours,
                'current_metrics': {
                    'timestamp': self.current_metrics.timestamp.isoformat(),
                    'events_per_second': self.current_metrics.events_per_second,
                    'processing_latency_ms': self.current_metrics.processing_latency_ms,
                    'queue_depth': self.current_metrics.queue_depth,
                    'active_alerts': self.current_metrics.active_alerts,
                    'system_cpu_percent': self.current_metrics.system_cpu_percent,
                    'memory_usage_percent': self.current_metrics.memory_usage_percent,
                    'disk_usage_percent': self.current_metrics.disk_usage_percent,
                    'network_io_mbps': self.current_metrics.network_io_mbps
                },
                'system_health': {
                    'overall_status': self.system_health.overall_status,
                    'last_check': self.system_health.last_check.isoformat(),
                    'issues': self.system_health.issues
                },
                'metrics_history': [
                    {
                        'timestamp': metrics.timestamp.isoformat(),
                        'events_per_second': metrics.events_per_second,
                        'processing_latency_ms': metrics.processing_latency_ms,
                        'queue_depth': metrics.queue_depth,
                        'active_alerts': metrics.active_alerts,
                        'system_cpu_percent': metrics.system_cpu_percent,
                        'memory_usage_percent': metrics.memory_usage_percent,
                        'disk_usage_percent': metrics.disk_usage_percent,
                        'network_io_mbps': metrics.network_io_mbps
                    }
                    for metrics in recent_metrics
                ],
                'websocket_connections': len(self.websocket_connections),
                'monitoring_active': self.monitoring_active
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Monitoring data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {str(e)}")
            return False