"""
Event Bus Architecture

Central event distribution system for real-time audit processing:
- Pub/Sub pattern for event distribution
- Event filtering and routing
- Priority-based event processing
- Event persistence and replay
- Fault-tolerant event handling
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json
import asyncio
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for audit system"""
    DECISION_MADE = "decision_made"
    DATA_ACCESS = "data_access"
    MODEL_PREDICTION = "model_prediction"
    USER_LOGIN = "user_login"
    CONFIG_CHANGE = "config_change"
    SYSTEM_ERROR = "system_error"
    PRIVACY_REQUEST = "privacy_request"
    CONSENT_CHANGE = "consent_change"
    ACCESS_DENIED = "access_denied"
    BREACH_DETECTED = "breach_detected"
    ANOMALY_DETECTED = "anomaly_detected"

class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class Event:
    """Audit event structure"""
    event_id: str
    event_type: EventType
    priority: EventPriority
    timestamp: datetime
    source: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'component': self.component,
            'data': self.data,
            'metadata': self.metadata,
            'correlation_id': self.correlation_id,
            'causation_id': self.causation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            priority=EventPriority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            component=data.get('component'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {}),
            correlation_id=data.get('correlation_id'),
            causation_id=data.get('causation_id')
        )

class EventFilter:
    """Event filter for subscription"""
    def __init__(self, 
                 event_types: Optional[List[EventType]] = None,
                 priorities: Optional[List[EventPriority]] = None,
                 sources: Optional[List[str]] = None,
                 components: Optional[List[str]] = None,
                 custom_filter: Optional[Callable[[Event], bool]] = None):
        self.event_types = set(event_types) if event_types else None
        self.priorities = set(priorities) if priorities else None
        self.sources = set(sources) if sources else None
        self.components = set(components) if components else None
        self.custom_filter = custom_filter
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter"""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.priorities and event.priority not in self.priorities:
            return False
        if self.sources and event.source not in self.sources:
            return False
        if self.components and event.component not in self.components:
            return False
        if self.custom_filter and not self.custom_filter(event):
            return False
        return True

class EventBus:
    """
    Central event bus for audit system
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._filtered_subscribers: List[tuple] = []  # (filter, callback)
        self._event_history: List[Event] = []
        self._max_history_size = 10000
        self._processing = False
        self._event_queue = asyncio.Queue()
        self._shutdown = False
        
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]) -> str:
        """Subscribe to specific event type"""
        subscription_id = str(uuid.uuid4())
        self._subscribers[event_type].append((subscription_id, callback))
        logger.info(f"Subscribed to {event_type.value} with ID {subscription_id}")
        return subscription_id
    
    def subscribe_filtered(self, event_filter: EventFilter, callback: Callable[[Event], None]) -> str:
        """Subscribe with custom filter"""
        subscription_id = str(uuid.uuid4())
        self._filtered_subscribers.append((subscription_id, event_filter, callback))
        logger.info(f"Subscribed with filter {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        # Remove from type-specific subscribers
        for event_type, subscribers in self._subscribers.items():
            self._subscribers[event_type] = [
                (sid, callback) for sid, callback in subscribers 
                if sid != subscription_id
            ]
        
        # Remove from filtered subscribers
        self._filtered_subscribers = [
            (sid, filter_obj, callback) for sid, filter_obj, callback in self._filtered_subscribers
            if sid != subscription_id
        ]
        
        logger.info(f"Unsubscribed {subscription_id}")
        return True
    
    async def publish(self, event: Event) -> None:
        """Publish event to all subscribers"""
        try:
            # Add to history
            self._add_to_history(event)
            
            # Add to processing queue
            await self._event_queue.put(event)
            
            # Log high priority events
            if event.priority.value >= EventPriority.HIGH.value:
                logger.warning(f"High priority event: {event.event_type.value} from {event.source}")
                
        except Exception as e:
            logger.error(f"Error publishing event {event.event_id}: {str(e)}")
    
    def publish_sync(self, event: Event) -> None:
        """Synchronous publish for non-async contexts"""
        asyncio.create_task(self.publish(event))
    
    async def start_processing(self) -> None:
        """Start event processing loop"""
        self._processing = True
        logger.info("Event bus processing started")
        
        while not self._shutdown:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                await self._process_event(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event processing loop: {str(e)}")
    
    async def stop_processing(self) -> None:
        """Stop event processing"""
        self._shutdown = True
        self._processing = False
        logger.info("Event bus processing stopped")
    
    async def _process_event(self, event: Event) -> None:
        """Process single event"""
        try:
            # Notify type-specific subscribers
            type_subscribers = self._subscribers.get(event.event_type, [])
            for subscription_id, callback in type_subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
                except Exception as e:
                    logger.error(f"Error in subscriber {subscription_id}: {str(e)}")
            
            # Notify filtered subscribers
            for subscription_id, event_filter, callback in self._filtered_subscribers:
                try:
                    if event_filter.matches(event):
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event)
                        else:
                            callback(event)
                except Exception as e:
                    logger.error(f"Error in filtered subscriber {subscription_id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {str(e)}")
    
    def _add_to_history(self, event: Event) -> None:
        """Add event to history with size limit"""
        self._event_history.append(event)
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]
    
    def get_events(self, 
                  event_type: Optional[EventType] = None,
                  source: Optional[str] = None,
                  user_id: Optional[str] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  limit: Optional[int] = None) -> List[Event]:
        """Get events with filtering"""
        events = self._event_history.copy()
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    def get_event_by_id(self, event_id: str) -> Optional[Event]:
        """Get specific event by ID"""
        for event in self._event_history:
            if event.event_id == event_id:
                return event
        return None
    
    def get_correlated_events(self, correlation_id: str) -> List[Event]:
        """Get all events with same correlation ID"""
        return [e for e in self._event_history if e.correlation_id == correlation_id]
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        if not self._event_history:
            return {}
        
        # Count by type
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for event in self._event_history:
            type_counts[event.event_type.value] += 1
            priority_counts[event.priority.value] += 1
            source_counts[event.source] += 1
        
        return {
            'total_events': len(self._event_history),
            'event_types': dict(type_counts),
            'priorities': dict(priority_counts),
            'sources': dict(source_counts),
            'subscribers': {
                'type_specific': sum(len(subs) for subs in self._subscribers.values()),
                'filtered': len(self._filtered_subscribers)
            },
            'processing': self._processing,
            'queue_size': self._event_queue.qsize()
        }
    
    def create_event(self, 
                   event_type: EventType,
                   source: str,
                   priority: EventPriority = EventPriority.MEDIUM,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   component: Optional[str] = None,
                   data: Optional[Dict[str, Any]] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   correlation_id: Optional[str] = None,
                   causation_id: Optional[str] = None) -> Event:
        """Create new event"""
        return Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            priority=priority,
            timestamp=datetime.now(),
            source=source,
            user_id=user_id,
            session_id=session_id,
            component=component,
            data=data or {},
            metadata=metadata or {},
            correlation_id=correlation_id,
            causation_id=causation_id
        )
    
    async def replay_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          event_types: Optional[List[EventType]] = None) -> int:
        """Replay events from history"""
        events = self.get_events(
            start_time=start_time,
            end_time=end_time
        )
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        # Sort by timestamp for proper replay order
        events.sort(key=lambda e: e.timestamp)
        
        replayed_count = 0
        for event in events:
            await self._process_event(event)
            replayed_count += 1
        
        logger.info(f"Replayed {replayed_count} events")
        return replayed_count
    
    def export_events(self, 
                     filepath: str,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     event_types: Optional[List[EventType]] = None) -> int:
        """Export events to file"""
        events = self.get_events(
            start_time=start_time,
            end_time=end_time
        )
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_events': len(events),
            'events': [event.to_dict() for event in events]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(events)} events to {filepath}")
        return len(events)
    
    def clear_history(self, older_than: Optional[datetime] = None) -> int:
        """Clear event history"""
        if older_than:
            original_count = len(self._event_history)
            self._event_history = [
                e for e in self._event_history if e.timestamp >= older_than
            ]
            removed_count = original_count - len(self._event_history)
        else:
            removed_count = len(self._event_history)
            self._event_history = []
        
        logger.info(f"Cleared {removed_count} events from history")
        return removed_count