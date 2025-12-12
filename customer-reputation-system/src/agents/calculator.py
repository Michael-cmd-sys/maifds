"""
Agent risk calculator for agent/merchant risk profiling
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.models import AgentRiskProfile, AgentRiskFactors, AgentNetworkMetrics
from src.mule_network.models import MuleAccount, NetworkRiskMetrics
from src.storage.database import DatabaseManager


class AgentRiskCalculator:
    """Calculate and manage agent risk profiles"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def calculate_agent_risk_factors(self, agent_id: str) -> AgentRiskFactors:
        """Calculate risk factors for an agent"""
        
        # Get agent data
        agent_data = self._get_agent_data(agent_id)
        if not agent_data:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Get network data
        network_data = self._get_agent_network_data(agent_id)
        
        # Calculate individual risk factors
        recruitment_velocity = self._calculate_recruitment_velocity(agent_data, network_data)
        network_growth_rate = self._calculate_network_growth_rate(network_data)
        transaction_anomaly_score = self._calculate_transaction_anomaly_score(network_data)
        geographic_dispersion = self._calculate_geographic_dispersion(network_data)
        temporal_patterns = self._calculate_temporal_patterns(network_data)
        communication_risk = self._calculate_communication_risk(agent_data)
        financial_behavior_score = self._calculate_financial_behavior_score(network_data)
        association_risk = self._calculate_association_risk(network_data)
        
        return AgentRiskFactors(
            recruitment_velocity=recruitment_velocity,
            network_growth_rate=network_growth_rate,
            transaction_anomaly_score=transaction_anomaly_score,
            geographic_dispersion=geographic_dispersion,
            temporal_patterns=temporal_patterns,
            communication_risk=communication_risk,
            financial_behavior_score=financial_behavior_score,
            association_risk=association_risk
        )

    def calculate_agent_risk_score(self, agent_id: str) -> float:
        """Calculate composite risk score for an agent"""
        risk_factors = self.calculate_agent_risk_factors(agent_id)
        return risk_factors.calculate_composite_risk()

    def update_agent_risk_profile(self, agent_id: str) -> AgentRiskProfile:
        """Update agent risk profile with latest calculations"""
        
        # Get current agent data
        agent_data = self._get_agent_data(agent_id)
        if not agent_data:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Calculate new risk score
        risk_score = self.calculate_agent_risk_score(agent_id)
        
        # Calculate credibility score (inverse of risk with some baseline)
        credibility_score = max(0.1, 1.0 - (risk_score * 0.8))
        
        # Get network metrics
        network_metrics = self._calculate_network_metrics(agent_id)
        
        # Update agent profile
        updated_profile = AgentRiskProfile(
            agent_id=agent_id,
            agent_name=agent_data.get("agent_name"),
            credibility_score=credibility_score,
            risk_score=risk_score,
            total_recruits=agent_data.get("total_recruits", 0),
            active_merchants=agent_data.get("active_merchants", 0),
            network_depth=network_metrics.total_nodes if network_metrics else 0,
            recruitment_rate=agent_data.get("recruitment_rate", 0.0),
            avg_transaction_amount=agent_data.get("avg_transaction_amount", 0.0),
            suspicious_activity_count=agent_data.get("suspicious_activity_count", 0),
            created_at=datetime.fromisoformat(agent_data["created_at"]) if agent_data.get("created_at") else None,
            updated_at=datetime.now()
        )
        
        # Save to database
        self._save_agent_profile(updated_profile)
        
        return updated_profile

    def get_high_risk_agents(self, threshold: float = 0.7) -> List[AgentRiskProfile]:
        """Get agents with risk score above threshold"""
        query = """
        SELECT agent_id, agent_name, credibility_score, risk_score, total_recruits,
               active_merchants, network_depth, recruitment_rate, avg_transaction_amount,
               suspicious_activity_count, created_at, updated_at
        FROM agents 
        WHERE risk_score >= ?
        ORDER BY risk_score DESC
        """
        
        rows = self.db.execute_query(query, (threshold,))
        agents = []
        
        for row in rows:
            agent = AgentRiskProfile(
                agent_id=row["agent_id"],
                agent_name=row["agent_name"],
                credibility_score=row["credibility_score"],
                risk_score=row["risk_score"],
                total_recruits=row["total_recruits"],
                active_merchants=row["active_merchants"],
                network_depth=row["network_depth"],
                recruitment_rate=row["recruitment_rate"],
                avg_transaction_amount=row["avg_transaction_amount"],
                suspicious_activity_count=row["suspicious_activity_count"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            )
            agents.append(agent)
        
        return agents

    def _get_agent_data(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent data from database"""
        query = """
        SELECT agent_id, agent_name, credibility_score, risk_score, total_recruits,
               active_merchants, network_depth, recruitment_rate, avg_transaction_amount,
               suspicious_activity_count, created_at, updated_at
        FROM agents WHERE agent_id = ?
        """
        
        rows = self.db.execute_query(query, (agent_id,))
        if not rows:
            return None
        
        row = rows[0]
        return {
            "agent_id": row["agent_id"],
            "agent_name": row["agent_name"],
            "credibility_score": row["credibility_score"],
            "risk_score": row["risk_score"],
            "total_recruits": row["total_recruits"],
            "active_merchants": row["active_merchants"],
            "network_depth": row["network_depth"],
            "recruitment_rate": row["recruitment_rate"],
            "avg_transaction_amount": row["avg_transaction_amount"],
            "suspicious_activity_count": row["suspicious_activity_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def _get_agent_network_data(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get network relationship data for agent"""
        query = """
        SELECT network_id, merchant_id, relationship_type, strength_score,
               transaction_count, total_amount, risk_level, first_interaction, last_interaction
        FROM agent_networks 
        WHERE agent_id = ?
        """
        
        rows = self.db.execute_query(query, (agent_id,))
        network_data = []
        
        for row in rows:
            network_data.append({
                "network_id": row["network_id"],
                "merchant_id": row["merchant_id"],
                "relationship_type": row["relationship_type"],
                "strength_score": row["strength_score"],
                "transaction_count": row["transaction_count"],
                "total_amount": row["total_amount"],
                "risk_level": row["risk_level"],
                "first_interaction": row["first_interaction"],
                "last_interaction": row["last_interaction"]
            })
        
        return network_data

    def _calculate_recruitment_velocity(self, agent_data: Dict[str, Any], 
                                     network_data: List[Dict[str, Any]]) -> float:
        """Calculate recruitment velocity score"""
        total_recruits = agent_data.get("total_recruits", 0)
        recruitment_rate = agent_data.get("recruitment_rate", 0.0)
        
        # Normalize recruitment rate (assuming 5 recruits/month is high)
        normalized_rate = min(1.0, recruitment_rate / 5.0)
        
        # Factor in total recruits (assuming 20+ is high)
        normalized_total = min(1.0, total_recruits / 20.0)
        
        return (normalized_rate * 0.7) + (normalized_total * 0.3)

    def _calculate_network_growth_rate(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate network growth rate score"""
        if not network_data:
            return 0.0
        
        # Calculate growth based on relationship creation dates
        now = datetime.now()
        recent_relationships = 0
        total_relationships = len(network_data)
        
        for rel in network_data:
            if rel.get("first_interaction"):
                first_date = datetime.fromisoformat(rel["first_interaction"])
                days_since_creation = (now - first_date).days
                if days_since_creation <= 30:  # Recent relationships
                    recent_relationships += 1
        
        # High growth if many recent relationships
        if total_relationships == 0:
            return 0.0
        
        growth_ratio = recent_relationships / total_relationships
        return min(1.0, growth_ratio * 3.0)  # Scale up for sensitivity

    def _calculate_transaction_anomaly_score(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate transaction anomaly score"""
        if not network_data:
            return 0.0
        
        high_risk_transactions = 0
        total_transactions = 0
        
        for rel in network_data:
            transaction_count = rel.get("transaction_count", 0)
            total_amount = rel.get("total_amount", 0.0)
            risk_level = rel.get("risk_level", "low")
            
            total_transactions += transaction_count
            
            # Flag high transaction counts or amounts
            if transaction_count > 100 or total_amount > 100000:
                high_risk_transactions += 1
            
            # Flag high risk relationships
            if risk_level == "high":
                high_risk_transactions += transaction_count * 0.5
        
        if total_transactions == 0:
            return 0.0
        
        anomaly_ratio = high_risk_transactions / total_transactions
        return min(1.0, anomaly_ratio * 2.0)

    def _calculate_geographic_dispersion(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate geographic dispersion score (simplified)"""
        # For now, use relationship diversity as proxy
        if not network_data:
            return 0.0
        
        relationship_types = set(rel.get("relationship_type", "") for rel in network_data)
        diversity_score = len(relationship_types) / 5.0  # Assuming 5 types max
        
        return min(1.0, diversity_score)

    def _calculate_temporal_patterns(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate temporal pattern risk score"""
        if not network_data:
            return 0.0
        
        # Check for unusual timing patterns
        unusual_patterns = 0
        total_relationships = len(network_data)
        
        for rel in network_data:
            if rel.get("first_interaction") and rel.get("last_interaction"):
                first_date = datetime.fromisoformat(rel["first_interaction"])
                last_date = datetime.fromisoformat(rel["last_interaction"])
                duration_days = (last_date - first_date).days
                
                # Flag very short or very long relationships
                if duration_days < 7 or duration_days > 365:
                    unusual_patterns += 1
        
        if total_relationships == 0:
            return 0.0
        
        pattern_ratio = unusual_patterns / total_relationships
        return min(1.0, pattern_ratio)

    def _calculate_communication_risk(self, agent_data: Dict[str, Any]) -> float:
        """Calculate communication risk score"""
        # Use suspicious activity count as proxy
        suspicious_count = agent_data.get("suspicious_activity_count", 0)
        return min(1.0, suspicious_count / 10.0)

    def _calculate_financial_behavior_score(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate financial behavior risk score"""
        if not network_data:
            return 0.0
        
        # Check for unusual financial patterns
        total_amount = sum(rel.get("total_amount", 0.0) for rel in network_data)
        avg_amount = total_amount / len(network_data) if network_data else 0.0
        
        # High amounts indicate risk
        return min(1.0, avg_amount / 50000.0)

    def _calculate_association_risk(self, network_data: List[Dict[str, Any]]) -> float:
        """Calculate association risk based on network connections"""
        if not network_data:
            return 0.0
        
        high_risk_associations = 0
        total_associations = len(network_data)
        
        for rel in network_data:
            risk_level = rel.get("risk_level", "low")
            strength_score = rel.get("strength_score", 0.0)
            
            # Flag high risk or strong associations
            if risk_level == "high" or strength_score > 0.8:
                high_risk_associations += 1
        
        if total_associations == 0:
            return 0.0
        
        association_ratio = high_risk_associations / total_associations
        return min(1.0, association_ratio)

    def _calculate_network_metrics(self, agent_id: str) -> Optional[AgentNetworkMetrics]:
        """Calculate network-level metrics for agent"""
        network_data = self._get_agent_network_data(agent_id)
        
        if not network_data:
            return None
        
        total_nodes = len(network_data)
        network_density = min(1.0, total_nodes / 50.0)  # Assuming 50 is dense network
        clustering_coefficient = 0.5  # Simplified
        average_path_length = 2.0  # Simplified
        centrality_score = min(1.0, total_nodes / 20.0)
        bridge_nodes_count = int(total_nodes * 0.1)  # Simplified
        community_count = max(1, total_nodes // 10)  # Simplified
        risk_propagation_score = self.calculate_agent_risk_score(agent_id)
        
        return AgentNetworkMetrics(
            total_nodes=total_nodes,
            network_density=network_density,
            clustering_coefficient=clustering_coefficient,
            average_path_length=average_path_length,
            centrality_score=centrality_score,
            bridge_nodes_count=bridge_nodes_count,
            community_count=community_count,
            risk_propagation_score=risk_propagation_score
        )

    def _save_agent_profile(self, profile: AgentRiskProfile) -> None:
        """Save agent profile to database"""
        query = """
        INSERT OR REPLACE INTO agents 
        (agent_id, agent_name, credibility_score, risk_score, total_recruits,
         active_merchants, network_depth, recruitment_rate, avg_transaction_amount,
         suspicious_activity_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        values = (
            profile.agent_id,
            profile.agent_name,
            profile.credibility_score,
            profile.risk_score,
            profile.total_recruits,
            profile.active_merchants,
            profile.network_depth,
            profile.recruitment_rate,
            profile.avg_transaction_amount,
            profile.suspicious_activity_count,
            profile.created_at.isoformat() if profile.created_at else None,
            profile.updated_at.isoformat() if profile.updated_at else None
        )
        
        self.db.execute_update(query, values)