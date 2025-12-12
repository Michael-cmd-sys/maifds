"""
Pydantic models for mule network detection
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class MuleAccount(BaseModel):
    """Mule account information"""

    account_id: str
    account_type: str = Field(description="Type of account (merchant, agent, individual)")
    mule_score: float = Field(ge=0.0, le=1.0, description="Mule probability score 0-1")
    network_id: Optional[str] = None
    transaction_patterns: Optional[str] = Field(None, description="JSON of transaction patterns")
    risk_indicators: Optional[str] = Field(None, description="JSON of risk indicators")
    is_confirmed_mule: bool = Field(default=False, description="Confirmed mule status")
    detection_date: Optional[datetime] = None
    rapid_transaction_count: int = Field(default=0, ge=0, description="Count of rapid transactions")
    circular_transaction_count: int = Field(default=0, ge=0, description="Count of circular transactions")
    avg_hold_time_minutes: float = Field(default=0.0, ge=0.0, description="Average time funds are held")
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
        if self.detection_date:
            data["detection_date"] = self.detection_date.isoformat()
        return data


class MuleRiskFactors(BaseModel):
    """Factors used in mule risk calculation"""

    rapid_transaction_score: float = Field(ge=0.0, le=1.0, description="Score based on transaction velocity")
    circular_transaction_score: float = Field(ge=0.0, le=1.0, description="Score based on circular transactions")
    short_hold_time_score: float = Field(ge=0.0, le=1.0, description="Score based on quick fund movement")
    network_centrality_score: float = Field(ge=0.0, le=1.0, description="Score based on network position")
    amount_anomaly_score: float = Field(ge=0.0, le=1.0, description="Score based on unusual amounts")
    temporal_pattern_score: float = Field(ge=0.0, le=1.0, description="Score based on timing patterns")
    geographic_anomaly_score: float = Field(ge=0.0, le=1.0, description="Score based on geographic patterns")
    behavioral_consistency_score: float = Field(ge=0.0, le=1.0, description="Score based on behavior changes")

    model_config = {"extra": "forbid"}

    def calculate_mule_probability(self) -> float:
        """Calculate mule probability from all factors"""
        weights = {
            "rapid_transaction_score": 0.20,
            "circular_transaction_score": 0.25,
            "short_hold_time_score": 0.20,
            "network_centrality_score": 0.15,
            "amount_anomaly_score": 0.10,
            "temporal_pattern_score": 0.05,
            "geographic_anomaly_score": 0.03,
            "behavioral_consistency_score": 0.02,
        }
        
        mule_probability = sum(
            getattr(self, factor) * weight 
            for factor, weight in weights.items()
        )
        
        return min(1.0, max(0.0, mule_probability))


class NetworkRelationship(BaseModel):
    """Relationship between entities in the network"""

    network_id: str
    agent_id: str
    merchant_id: str
    relationship_type: str = Field(description="Type of relationship (recruited, associated, transactional)")
    strength_score: float = Field(ge=0.0, le=1.0, description="Strength of relationship")
    transaction_count: int = Field(default=0, ge=0, description="Number of transactions")
    total_amount: float = Field(default=0.0, ge=0.0, description="Total transaction amount")
    risk_level: str = Field(default="medium", description="Risk level (low, medium, high)")
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = {"extra": "forbid"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.model_dump()
        if self.first_interaction:
            data["first_interaction"] = self.first_interaction.isoformat()
        if self.last_interaction:
            data["last_interaction"] = self.last_interaction.isoformat()
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        return data


class NetworkRiskMetrics(BaseModel):
    """Network-level risk metrics"""

    network_id: str
    total_nodes: int = Field(ge=0, description="Total nodes in network")
    mule_density: float = Field(ge=0.0, le=1.0, description="Proportion of mule accounts")
    transaction_velocity: float = Field(ge=0.0, description="Transactions per day")
    avg_transaction_amount: float = Field(ge=0.0, description="Average transaction amount")
    network_risk_score: float = Field(ge=0.0, le=1.0, description="Overall network risk score")
    centralization_index: float = Field(ge=0.0, le=1.0, description="Network centralization")
    community_count: int = Field(ge=0, description="Number of distinct communities")
    bridge_edges_count: int = Field(ge=0, description="Number of bridge edges")
    last_updated: Optional[datetime] = None

    model_config = {"extra": "forbid"}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = self.model_dump()
        if self.last_updated:
            data["last_updated"] = self.last_updated.isoformat()
        return data