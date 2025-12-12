"""
Real-time risk scoring API for agent/merchant/mule network analysis
"""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from ..storage.database import DatabaseManager


class RealTimeRiskAPI:
    """Real-time API for risk scoring and fraud detection"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_agent_risk_score(self, agent_id: str) -> Dict[str, Any]:
        """Get real-time risk score for an agent"""
        
        # Get agent data
        query = """
        SELECT agent_id, agent_name, credibility_score, risk_score, total_recruits,
               active_merchants, network_depth, recruitment_rate, avg_transaction_amount,
               suspicious_activity_count, updated_at
        FROM agents WHERE agent_id = ?
        """
        
        rows = self.db.execute_query(query, (agent_id,))
        if not rows:
            return {"error": f"Agent {agent_id} not found"}
        
        agent_data = rows[0]
        
        # Get recent activity
        recent_activity = self._get_agent_recent_activity(agent_id)
        
        # Calculate dynamic risk factors
        dynamic_risk = self._calculate_dynamic_agent_risk(agent_id, recent_activity)
        
        return {
            "agent_id": agent_data["agent_id"],
            "agent_name": agent_data["agent_name"],
            "static_risk_score": agent_data["risk_score"],
            "dynamic_risk_score": dynamic_risk["risk_score"],
            "combined_risk_score": (agent_data["risk_score"] + dynamic_risk["risk_score"]) / 2,
            "credibility_score": agent_data["credibility_score"],
            "risk_level": self._get_risk_level((agent_data["risk_score"] + dynamic_risk["risk_score"]) / 2),
            "last_updated": agent_data["updated_at"],
            "recent_activity": recent_activity,
            "risk_factors": dynamic_risk["factors"],
            "recommendations": self._get_agent_recommendations(agent_data, dynamic_risk)
        }

    def get_merchant_risk_assessment(self, merchant_id: str) -> Dict[str, Any]:
        """Get comprehensive risk assessment for a merchant"""
        
        # Get merchant data
        query = """
        SELECT merchant_id, merchant_name, total_reports, average_rating, 
               reputation_score, updated_at
        FROM merchants WHERE merchant_id = ?
        """
        
        rows = self.db.execute_query(query, (merchant_id,))
        if not rows:
            return {"error": f"Merchant {merchant_id} not found"}
        
        merchant_data = rows[0]
        
        # Get agent connections
        agent_connections = self._get_merchant_agent_connections(merchant_id)
        
        # Get mule associations
        mule_associations = self._get_merchant_mule_associations(merchant_id)
        
        # Calculate merchant risk score
        merchant_risk = self._calculate_merchant_risk_score(
            merchant_data, agent_connections, mule_associations
        )
        
        return {
            "merchant_id": merchant_data["merchant_id"],
            "merchant_name": merchant_data["merchant_name"],
            "reputation_score": merchant_data["reputation_score"],
            "merchant_risk_score": merchant_risk["risk_score"],
            "risk_level": self._get_risk_level(merchant_risk["risk_score"]),
            "agent_connections": agent_connections,
            "mule_associations": mule_associations,
            "risk_factors": merchant_risk["factors"],
            "recommendations": self._get_merchant_recommendations(merchant_data, merchant_risk),
            "last_updated": merchant_data["updated_at"]
        }

    def detect_suspicious_transactions(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Detect suspicious transactions within time window"""
        
        # Get recent transactions
        query = """
        SELECT agent_id, merchant_id, transaction_count, total_amount, 
               strength_score, risk_level, last_interaction
        FROM agent_networks 
        WHERE last_interaction >= datetime('now', '-{} hours')
        ORDER BY last_interaction DESC
        """.format(time_window_hours)
        
        recent_transactions = self.db.execute_query(query)
        
        # Analyze for suspicious patterns
        suspicious_transactions = []
        
        for tx in recent_transactions:
            suspicion_score = self._calculate_transaction_suspicion(tx)
            if suspicion_score > 0.6:  # Threshold for suspicious
                suspicious_transactions.append({
                    "agent_id": tx["agent_id"],
                    "merchant_id": tx["merchant_id"],
                    "transaction_count": tx["transaction_count"],
                    "total_amount": tx["total_amount"],
                    "suspicion_score": suspicion_score,
                    "risk_level": tx["risk_level"],
                    "last_interaction": tx["last_interaction"],
                    "suspicion_factors": self._get_suspicion_factors(tx)
                })
        
        return {
            "time_window_hours": time_window_hours,
            "total_transactions_analyzed": len(recent_transactions),
            "suspicious_transactions_found": len(suspicious_transactions),
            "suspicious_transactions": suspicious_transactions[:50],  # Limit results
            "high_risk_alerts": [tx for tx in suspicious_transactions if tx["suspicion_score"] > 0.8]
        }

    def get_network_risk_overview(self, network_id: str = None) -> Dict[str, Any]:
        """Get overview of network risk"""
        
        if network_id:
            # Specific network analysis
            return self._get_specific_network_overview(network_id)
        else:
            # Global network overview
            return self._get_global_network_overview()

    def update_risk_scores_realtime(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Update risk scores in real-time based on new data"""
        
        try:
            if entity_type == "agent":
                # Import here to avoid circular imports
                from .agents.calculator import AgentRiskCalculator
                calculator = AgentRiskCalculator(self.db)
                updated_profile = calculator.update_agent_risk_profile(entity_id)
                
                return {
                    "success": True,
                    "entity_type": "agent",
                    "entity_id": entity_id,
                    "previous_risk_score": updated_profile.risk_score,
                    "new_risk_score": updated_profile.risk_score,
                    "credibility_score": updated_profile.credibility_score,
                    "updated_at": updated_profile.updated_at.isoformat() if updated_profile.updated_at else None
                }
            
            elif entity_type == "mule":
                # Update mule score based on new transaction data
                new_score = self._calculate_realtime_mule_score(entity_id)
                
                query = """
                UPDATE mule_accounts 
                SET mule_score = ?, updated_at = ?
                WHERE account_id = ?
                """
                
                success = self.db.execute_update(query, (new_score, datetime.now().isoformat(), entity_id))
                
                return {
                    "success": success,
                    "entity_type": "mule",
                    "entity_id": entity_id,
                    "new_mule_score": new_score,
                    "updated_at": datetime.now().isoformat()
                }
            
            else:
                return {"error": f"Unsupported entity type: {entity_type}"}
                
        except Exception as e:
            return {"error": f"Failed to update risk score: {str(e)}"}

    def get_risk_alerts(self, severity_threshold: float = 0.8) -> Dict[str, Any]:
        """Get current risk alerts"""
        
        alerts = []
        
        # High-risk agents
        high_risk_agents = self._get_high_risk_entities("agents", severity_threshold)
        alerts.extend([
            {
                "alert_type": "high_risk_agent",
                "entity_id": agent["agent_id"],
                "entity_name": agent["agent_name"],
                "risk_score": agent["risk_score"],
                "severity": "critical" if agent["risk_score"] > 0.9 else "high",
                "timestamp": datetime.now().isoformat(),
                "description": f"Agent {agent['agent_name']} has high risk score of {agent['risk_score']:.3f}"
            }
            for agent in high_risk_agents
        ])
        
        # High-risk mule accounts
        high_risk_mules = self._get_high_risk_entities("mule_accounts", severity_threshold)
        alerts.extend([
            {
                "alert_type": "high_risk_mule",
                "entity_id": mule["account_id"],
                "entity_type": mule["account_type"],
                "risk_score": mule["mule_score"],
                "severity": "critical" if mule["mule_score"] > 0.9 else "high",
                "timestamp": datetime.now().isoformat(),
                "description": f"Mule account {mule['account_id']} has high risk score of {mule['mule_score']:.3f}"
            }
            for mule in high_risk_mules
        ])
        
        # Suspicious transaction patterns
        suspicious_tx = self.detect_suspicious_transactions(1)  # Last hour
        alerts.extend([
            {
                "alert_type": "suspicious_transaction",
                "entity_id": tx["agent_id"],
                "related_entity": tx["merchant_id"],
                "suspicion_score": tx["suspicion_score"],
                "severity": "critical" if tx["suspicion_score"] > 0.9 else "high",
                "timestamp": tx["last_interaction"],
                "description": f"Suspicious transaction pattern detected: {tx['suspicion_score']:.3f} suspicion score"
            }
            for tx in suspicious_tx["high_risk_alerts"]
        ])
        
        # Sort by severity and timestamp
        alerts.sort(key=lambda x: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 4),
            x["timestamp"]
        ), reverse=True)
        
        return {
            "total_alerts": len(alerts),
            "severity_threshold": severity_threshold,
            "alerts": alerts[:100],  # Limit results
            "alert_summary": self._summarize_alerts(alerts)
        }

    def _get_agent_recent_activity(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get recent activity for an agent"""
        
        query = """
        SELECT merchant_id, transaction_count, total_amount, last_interaction
        FROM agent_networks 
        WHERE agent_id = ? AND last_interaction >= datetime('now', '-7 days')
        ORDER BY last_interaction DESC
        LIMIT 10
        """
        
        return self.db.execute_query(query, (agent_id,))

    def _calculate_dynamic_agent_risk(self, agent_id: str, recent_activity: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate dynamic risk based on recent activity"""
        
        if not recent_activity:
            return {"risk_score": 0.0, "factors": {}}
        
        # Risk factors
        high_frequency = min(1.0, sum(tx["transaction_count"] for tx in recent_activity) / 100.0)
        high_amount = min(1.0, sum(tx["total_amount"] for tx in recent_activity) / 100000.0)
        rapid_succession = min(1.0, len([tx for tx in recent_activity if self._is_rapid_transaction(tx)]) / len(recent_activity))
        
        dynamic_risk = (high_frequency * 0.4) + (high_amount * 0.4) + (rapid_succession * 0.2)
        
        return {
            "risk_score": dynamic_risk,
            "factors": {
                "high_frequency": high_frequency,
                "high_amount": high_amount,
                "rapid_succession": rapid_succession
            }
        }

    def _get_merchant_agent_connections(self, merchant_id: str) -> List[Dict[str, Any]]:
        """Get agent connections for a merchant"""
        
        query = """
        SELECT a.agent_id, a.agent_name, a.risk_score, an.transaction_count,
               an.total_amount, an.strength_score, an.risk_level
        FROM agents a
        JOIN agent_networks an ON a.agent_id = an.agent_id
        WHERE an.merchant_id = ?
        ORDER BY an.total_amount DESC
        LIMIT 20
        """
        
        return self.db.execute_query(query, (merchant_id,))

    def _get_merchant_mule_associations(self, merchant_id: str) -> List[Dict[str, Any]]:
        """Get mule associations for a merchant"""
        
        query = """
        SELECT ma.account_id, ma.account_type, ma.mule_score, an.transaction_count
        FROM mule_accounts ma
        JOIN agent_networks an ON ma.network_id = an.network_id
        WHERE an.merchant_id = ?
        ORDER BY ma.mule_score DESC
        LIMIT 10
        """
        
        return self.db.execute_query(query, (merchant_id,))

    def _calculate_merchant_risk_score(self, merchant_data: Dict[str, Any], 
                                   agent_connections: List[Dict[str, Any]], 
                                   mule_associations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive merchant risk score"""
        
        # Risk factors
        high_risk_agents = len([conn for conn in agent_connections if conn["risk_score"] > 0.7])
        agent_risk_factor = min(1.0, high_risk_agents / max(1, len(agent_connections)))
        
        high_risk_mules = len([assoc for assoc in mule_associations if assoc["mule_score"] > 0.7])
        mule_risk_factor = min(1.0, high_risk_mules / max(1, len(mule_associations)))
        
        reputation_risk = 1.0 - merchant_data["reputation_score"]
        
        merchant_risk = (agent_risk_factor * 0.4) + (mule_risk_factor * 0.4) + (reputation_risk * 0.2)
        
        return {
            "risk_score": merchant_risk,
            "factors": {
                "high_risk_agents": agent_risk_factor,
                "high_risk_mules": mule_risk_factor,
                "reputation_risk": reputation_risk
            }
        }

    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        elif risk_score >= 0.2:
            return "low"
        else:
            return "minimal"

    def _get_agent_recommendations(self, agent_data: Dict[str, Any], dynamic_risk: Dict[str, Any]) -> List[str]:
        """Get recommendations for agent risk mitigation"""
        
        recommendations = []
        combined_risk = (agent_data["risk_score"] + dynamic_risk["risk_score"]) / 2
        
        if combined_risk > 0.7:
            recommendations.append("Immediate investigation recommended due to high risk score")
            recommendations.append("Consider temporary suspension of recruitment activities")
        
        if dynamic_risk["factors"].get("high_frequency", 0) > 0.6:
            recommendations.append("Monitor transaction frequency - unusual activity detected")
        
        if dynamic_risk["factors"].get("high_amount", 0) > 0.6:
            recommendations.append("Review large transactions for compliance")
        
        if agent_data["network_depth"] > 4:
            recommendations.append("Analyze network structure for potential money laundering")
        
        return recommendations

    def _get_merchant_recommendations(self, merchant_data: Dict[str, Any], 
                                   merchant_risk: Dict[str, Any]) -> List[str]:
        """Get recommendations for merchant risk mitigation"""
        
        recommendations = []
        
        if merchant_risk["risk_score"] > 0.6:
            recommendations.append("Enhanced monitoring of transactions recommended")
            recommendations.append("Review agent relationships for suspicious patterns")
        
        if merchant_risk["factors"].get("high_risk_agents", 0) > 0.5:
            recommendations.append("Consider terminating high-risk agent relationships")
        
        if merchant_risk["factors"].get("high_risk_mules", 0) > 0.3:
            recommendations.append("Investigate potential money laundering connections")
        
        if merchant_data["total_reports"] > 10:
            recommendations.append("Review customer complaints and fraud reports")
        
        return recommendations

    def _calculate_transaction_suspicion(self, transaction: Dict[str, Any]) -> float:
        """Calculate suspicion score for a transaction"""
        
        # Risk factors
        high_amount = min(1.0, transaction["total_amount"] / 50000.0)
        high_frequency = min(1.0, transaction["transaction_count"] / 50.0)
        high_risk_level = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(transaction["risk_level"], 0.5)
        high_strength = transaction.get("strength_score", 0)
        
        return (high_amount * 0.3) + (high_frequency * 0.3) + (high_risk_level * 0.2) + (high_strength * 0.2)

    def _get_suspicion_factors(self, transaction: Dict[str, Any]) -> Dict[str, float]:
        """Get detailed suspicion factors for a transaction"""
        
        return {
            "high_amount": min(1.0, transaction["total_amount"] / 50000.0),
            "high_frequency": min(1.0, transaction["transaction_count"] / 50.0),
            "high_risk_level": {"low": 0.2, "medium": 0.5, "high": 0.8}.get(transaction["risk_level"], 0.5),
            "high_strength": transaction.get("strength_score", 0)
        }

    def _is_rapid_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Check if transaction shows rapid succession pattern"""
        
        # Simplified - would need timing data for accurate detection
        return transaction["transaction_count"] > 20

    def _get_high_risk_entities(self, table: str, threshold: float) -> List[Dict[str, Any]]:
        """Get high-risk entities from specified table"""
        
        if table == "agents":
            query = "SELECT agent_id, agent_name, risk_score FROM agents WHERE risk_score >= ? ORDER BY risk_score DESC LIMIT 20"
        elif table == "mule_accounts":
            query = "SELECT account_id, account_type, mule_score FROM mule_accounts WHERE mule_score >= ? ORDER BY mule_score DESC LIMIT 20"
        else:
            return []
        
        return self.db.execute_query(query, (threshold,))

    def _get_specific_network_overview(self, network_id: str) -> Dict[str, Any]:
        """Get overview for specific network"""
        
        query = """
        SELECT COUNT(DISTINCT agent_id) as agent_count,
               COUNT(DISTINCT merchant_id) as merchant_count,
               COUNT(*) as total_relationships,
               SUM(transaction_count) as total_transactions,
               SUM(total_amount) as total_amount
        FROM agent_networks 
        WHERE network_id = ?
        """
        
        rows = self.db.execute_query(query, (network_id,))
        if not rows:
            return {"error": f"Network {network_id} not found"}
        
        network_data = rows[0]
        
        return {
            "network_id": network_id,
            "agent_count": network_data["agent_count"],
            "merchant_count": network_data["merchant_count"],
            "total_relationships": network_data["total_relationships"],
            "total_transactions": network_data["total_transactions"] or 0,
            "total_amount": network_data["total_amount"] or 0,
            "avg_transaction_amount": (network_data["total_amount"] or 0) / max(1, network_data["total_transactions"] or 1),
            "network_size": "small" if network_data["total_relationships"] < 10 else "medium" if network_data["total_relationships"] < 50 else "large"
        }

    def _get_global_network_overview(self) -> Dict[str, Any]:
        """Get global network overview"""
        
        queries = {
            "total_agents": "SELECT COUNT(*) as count FROM agents",
            "total_merchants": "SELECT COUNT(*) as count FROM merchants",
            "total_mule_accounts": "SELECT COUNT(*) as count FROM mule_accounts",
            "total_relationships": "SELECT COUNT(*) as count FROM agent_networks",
            "high_risk_agents": "SELECT COUNT(*) as count FROM agents WHERE risk_score > 0.7",
            "high_risk_mules": "SELECT COUNT(*) as count FROM mule_accounts WHERE mule_score > 0.7"
        }
        
        overview = {}
        for key, query in queries.items():
            rows = self.db.execute_query(query)
            overview[key] = rows[0]["count"] if rows else 0
        
        # Calculate derived metrics
        overview["high_risk_agent_percentage"] = (overview["high_risk_agents"] / max(1, overview["total_agents"])) * 100
        overview["high_risk_mule_percentage"] = (overview["high_risk_mules"] / max(1, overview["total_mule_accounts"])) * 100
        
        return overview

    def _calculate_realtime_mule_score(self, account_id: str) -> float:
        """Calculate real-time mule score based on recent activity"""
        
        # Get recent transaction patterns
        query = """
        SELECT transaction_count, total_amount, last_interaction
        FROM agent_networks an
        JOIN mule_accounts ma ON ma.network_id = an.network_id
        WHERE ma.account_id = ? AND last_interaction >= datetime('now', '-24 hours')
        """
        
        recent_activity = self.db.execute_query(query, (account_id,))
        
        if not recent_activity:
            return 0.0
        
        # Calculate risk based on recent activity
        total_tx = sum(tx["transaction_count"] for tx in recent_activity)
        total_amount = sum(tx["total_amount"] for tx in recent_activity)
        
        frequency_risk = min(1.0, total_tx / 50.0)
        amount_risk = min(1.0, total_amount / 25000.0)
        
        return (frequency_risk + amount_risk) / 2

    def _summarize_alerts(self, alerts: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize alerts by type and severity"""
        
        summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "by_type": {}
        }
        
        for alert in alerts:
            severity = alert.get("severity", "low")
            summary[severity] = summary.get(severity, 0) + 1
            
            alert_type = alert.get("alert_type", "unknown")
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1
        
        return summary