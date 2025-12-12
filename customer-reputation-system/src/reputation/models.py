"""
Pydantic models for merchant reputation
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class MerchantReputation(BaseModel):
    """Merchant reputation information"""

    merchant_id: str
    reputation_score: float = Field(ge=0.0, le=1.0, description="Reputation score 0-1")
    total_reports: int = Field(ge=0)
    average_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    credibility_weighted_rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    positive_reports_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    negative_reports_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    fraud_reports_ratio: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_trend: Optional[str] = Field(None, description="trending_up, trending_down, stable")
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


class ReputationFactors(BaseModel):
    """Factors used in reputation calculation"""

    weighted_rating_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=0.0, le=1.0)
    fraud_risk_score: float = Field(ge=0.0, le=1.0)
    report_volume_score: float = Field(ge=0.0, le=1.0)
    time_decay_factor: float = Field(ge=0.0, le=1.0)
    credibility_weight: float = Field(ge=0.0, le=1.0, description="Average reporter credibility")

    model_config = {"extra": "forbid"}

