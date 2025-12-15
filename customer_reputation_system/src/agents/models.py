"""
Pydantic models for agent risk profiling
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class AgentRiskProfile(BaseModel):
    """Agent risk profile information"""

    agent_id: str
    agent_name: Optional[str] = None
    credibility_score: float = Field(ge=0.0, le=1.0, description="Agent credibility score 0-1")
    risk_score: float = Field(ge=0.0, le=1.0, description="Agent risk score 0-1")
    total_recruits: int = Field(ge=0, description="Total number of recruited merchants")
    active_merchants: int = Field(ge=0, description="Currently active merchants")
    network_depth: int = Field(ge=0, description="Depth of agent network")
    recruitment_rate: float = Field(ge=0.0, description="Merchants recruited per month")
    avg_transaction_amount: float = Field(ge=0.0, description="Average transaction amount")
    suspicious_activity_count: int = Field(ge=0, description="Count of suspicious activities")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"extra": "forbid"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.model_dump()
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data


class AgentRiskFactors(BaseModel):
    """Factors used in agent risk calculation"""

    recruitment_velocity: float = Field(ge=0.0, le=1.0, description="Speed of merchant recruitment")
    network_growth_rate: float = Field(ge=0.0, le=1.0, description="Network expansion speed")
    transaction_anomaly_score: float = Field(ge=0.0, le=1.0, description="Unusual transaction patterns")
    geographic_dispersion: float = Field(ge=0.0, le=1.0, description="Geographic spread of network")
    temporal_patterns: float = Field(ge=0.0, le=1.0, description="Suspicious timing patterns")
    communication_risk: float = Field(ge=0.0, le=1.0, description="Risky communication indicators")
    financial_behavior_score: float = Field(ge=0.0, le=1.0, description="Financial behavior risk")
    association_risk: float = Field(ge=0.0, le=1.0, description="Risk from known associations")

    model_config = {"extra": "forbid"}

    def calculate_composite_risk(self) -> float:
        """Calculate composite risk score from all factors"""
        weights = {
            "recruitment_velocity": 0.15,
            "network_growth_rate": 0.15,
            "transaction_anomaly_score": 0.20,
            "geographic_dispersion": 0.10,
            "temporal_patterns": 0.15,
            "communication_risk": 0.10,
            "financial_behavior_score": 0.10,
            "association_risk": 0.05,
        }
        
        composite_score = sum(
            getattr(self, factor) * weight 
            for factor, weight in weights.items()
        )
        
        return min(1.0, max(0.0, composite_score))


class AgentNetworkMetrics(BaseModel):
    """Network-level metrics for agent analysis"""

    total_nodes: int = Field(ge=0, description="Total nodes in network")
    network_density: float = Field(ge=0.0, le=1.0, description="Network connectivity density")
    clustering_coefficient: float = Field(ge=0.0, le=1.0, description="Network clustering")
    average_path_length: float = Field(ge=0.0, description="Average path between nodes")
    centrality_score: float = Field(ge=0.0, le=1.0, description="Agent's centrality in network")
    bridge_nodes_count: int = Field(ge=0, description="Number of bridge nodes")
    community_count: int = Field(ge=0, description="Number of distinct communities")
    risk_propagation_score: float = Field(ge=0.0, le=1.0, description="Risk propagation potential")

    model_config = {"extra": "forbid"}