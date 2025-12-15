"""
Pydantic models for reporter credibility
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class ReporterCredibility(BaseModel):
    """Reporter credibility information"""

    reporter_id: str
    credibility_score: float = Field(ge=0.0, le=1.0, description="Credibility score 0-1")
    total_reports: int = Field(ge=0)
    verified_reports: int = Field(ge=0)
    average_text_credibility: Optional[float] = Field(None, ge=0.0, le=1.0)
    consistency_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_activity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
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


class CredibilityFactors(BaseModel):
    """Factors used in credibility calculation"""

    text_credibility_score: float = Field(ge=0.0, le=1.0)
    report_consistency: float = Field(ge=0.0, le=1.0)
    verification_rate: float = Field(ge=0.0, le=1.0)
    time_decay_factor: float = Field(ge=0.0, le=1.0)
    report_quality_metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

