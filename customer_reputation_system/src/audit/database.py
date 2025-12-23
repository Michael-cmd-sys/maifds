"""
Database operations for audit trail and compliance
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from customer_reputation_system.config.logging_config import setup_logger
from schemas import ALL_AUDIT_SCHEMAS
from models import (
    AuditEvent, DecisionEvent, DataAccessEvent, 
    ModelPredictionEvent, PrivacyRequestEvent
)

logger = setup_logger(__name__)


class AuditDatabaseManager:
    """Manages audit database connections and operations"""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize audit database manager"""
        if db_path is None:
            db_path = Path("data/audit.db")
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self):
        """Create audit tables and indexes if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for schema in ALL_AUDIT_SCHEMAS:
                    cursor.execute(schema)
                conn.commit()
                logger.info(f"Audit database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def store_audit_event(self, event: AuditEvent) -> bool:
        """
        Store an audit event in the database
        
        Args:
            event: AuditEvent object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    INSERT INTO audit_events (
                        event_id, timestamp, event_type, component, user_id, session_id,
                        entity_id, entity_type, decision_data, explanation_data,
                        privacy_impact, ip_address, user_agent, success, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_id,
                        event.timestamp.isoformat(),
                        event.event_type,
                        event.component,
                        event.user_id,
                        event.session_id,
                        event.entity_id,
                        event.entity_type,
                        json.dumps(event.decision_data) if event.decision_data else None,
                        json.dumps(event.explanation_data) if event.explanation_data else None,
                        event.privacy_impact,
                        event.ip_address,
                        event.user_agent,
                        event.success,
                        event.error_message,
                        json.dumps(event.metadata) if event.metadata else None
                    ),
                )
                conn.commit()
                logger.debug(f"Audit event stored: {event.event_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store audit event {event.event_id}: {e}")
            return False

    def store_decision_event(self, decision_event: DecisionEvent) -> bool:
        """Store a decision event"""
        audit_event = decision_event.to_audit_event()
        return self.store_audit_event(audit_event)

    def store_data_access_event(self, access_event: DataAccessEvent) -> bool:
        """Store a data access event"""
        audit_event = access_event.to_audit_event()
        return self.store_audit_event(audit_event)

    def store_model_prediction_event(self, prediction_event: ModelPredictionEvent) -> bool:
        """Store a model prediction event"""
        audit_event = prediction_event.to_audit_event()
        return self.store_audit_event(audit_event)

    def store_privacy_request_event(self, privacy_event: PrivacyRequestEvent) -> bool:
        """Store a privacy request event"""
        audit_event = privacy_event.to_audit_event()
        return self.store_audit_event(audit_event)

    def get_audit_events(
        self, 
        limit: int = 1000,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit events with optional filters
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            component: Filter by component
            user_id: Filter by user ID
            start_date: Filter events after this date
            end_date: Filter events before this date
            
        Returns:
            List of audit event dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                    
                if component:
                    query += " AND component = ?"
                    params.append(component)
                    
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                    
                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())
                    
                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve audit events: {e}")
            return []

    def get_audit_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific audit event by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audit_events WHERE event_id = ?",
                    (event_id,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve audit event {event_id}: {e}")
            return None

    def get_user_audit_trail(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get audit trail for a specific user"""
        start_date = datetime.now() - timedelta(days=days)
        return self.get_audit_events(
            user_id=user_id,
            start_date=start_date,
            limit=1000
        )

    def get_entity_audit_trail(self, entity_id: str, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit trail for a specific entity"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM audit_events WHERE entity_id = ?"
                params = [entity_id]
                
                if entity_type:
                    query += " AND entity_type = ?"
                    params.append(entity_type)
                    
                query += " ORDER BY timestamp DESC LIMIT 1000"
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to retrieve entity audit trail: {e}")
            return []

    def get_privacy_impact_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of privacy impact events"""
        start_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT privacy_impact, COUNT(*) as count
                    FROM audit_events 
                    WHERE timestamp >= ? AND privacy_impact != 'none'
                    GROUP BY privacy_impact
                    ORDER BY count DESC
                """,
                    (start_date.isoformat(),)
                )
                
                impact_summary = {}
                for row in cursor.fetchall():
                    impact_summary[row['privacy_impact']] = row['count']
                
                return impact_summary
                
        except Exception as e:
            logger.error(f"Failed to get privacy impact summary: {e}")
            return {}

    def get_component_activity(self, component: str, days: int = 7) -> Dict[str, Any]:
        """Get activity summary for a component"""
        start_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM audit_events 
                    WHERE component = ? AND timestamp >= ?
                    GROUP BY event_type
                    ORDER BY count DESC
                """,
                    (component, start_date.isoformat())
                )
                
                activity_summary = {}
                for row in cursor.fetchall():
                    activity_summary[row['event_type']] = row['count']
                
                return activity_summary
                
        except Exception as e:
            logger.error(f"Failed to get component activity: {e}")
            return {}

    def export_audit_trail(
        self, 
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Export audit trail for compliance reporting
        
        Args:
            format: Export format (json, csv)
            start_date: Start date for export
            end_date: End date for export
            event_types: List of event types to include
            
        Returns:
            Dictionary with export data and metadata
        """
        try:
            events = self.get_audit_events(
                limit=100000,  # Large limit for export
                start_date=start_date,
                end_date=end_date
            )
            
            # Filter by event types if specified
            if event_types:
                events = [e for e in events if e.get('event_type') in event_types]
            
            export_data = {
                "events": events,
                "export_metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_events": len(events),
                    "date_range": {
                        "start": start_date.isoformat() if start_date else None,
                        "end": end_date.isoformat() if end_date else None
                    },
                    "event_types": event_types,
                    "format": format
                }
            }
            
            if format.lower() == "csv":
                # Convert to CSV format
                import csv
                import io
                
                output = io.StringIO()
                if events:
                    writer = csv.DictWriter(output, fieldnames=events[0].keys())
                    writer.writeheader()
                    writer.writerows(events)
                
                export_data["csv_data"] = output.getvalue()
            else:
                # JSON format (default)
                export_data["json_data"] = json.dumps(events, indent=2, default=str)
            
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return {"error": str(e)}

    def cleanup_old_events(self, retention_days: int = 365) -> int:
        """
        Clean up old audit events based on retention policy
        
        Args:
            retention_days: Number of days to retain events
            
        Returns:
            Number of events deleted
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute(
                    "DELETE FROM audit_events WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old audit events")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup old events: {e}")
            return 0

    def get_audit_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        start_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total events
                cursor.execute(
                    "SELECT COUNT(*) as total FROM audit_events WHERE timestamp >= ?",
                    (start_date.isoformat(),)
                )
                total_events = cursor.fetchone()["total"]
                
                # Events by type
                cursor.execute(
                    """
                    SELECT event_type, COUNT(*) as count
                    FROM audit_events 
                    WHERE timestamp >= ?
                    GROUP BY event_type
                    ORDER BY count DESC
                """,
                    (start_date.isoformat(),)
                )
                events_by_type = dict(cursor.fetchall())
                
                # Events by component
                cursor.execute(
                    """
                    SELECT component, COUNT(*) as count
                    FROM audit_events 
                    WHERE timestamp >= ?
                    GROUP BY component
                    ORDER BY count DESC
                """,
                    (start_date.isoformat(),)
                )
                events_by_component = dict(cursor.fetchall())
                
                # Privacy impact summary
                privacy_impact = self.get_privacy_impact_summary(days)
                
                return {
                    "period_days": days,
                    "total_events": total_events,
                    "events_by_type": events_by_type,
                    "events_by_component": events_by_component,
                    "privacy_impact_summary": privacy_impact,
                    "generated_at": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}