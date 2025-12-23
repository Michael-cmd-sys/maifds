"""
Pydantic models for report data validation
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import Optional, Literal, Dict, Any
from uuid import uuid4
import re


class ReportMetadata(BaseModel):
    """Metadata associated with a report"""

    platform: Literal["mobile", "web", "api"]
    location: Optional[str] = None
    device_info: Optional[str] = None
    # ✅ allow NLP analysis to be stored without breaking reads
    nlp_analysis: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid")


class Report(BaseModel):
    """Main report model with validation"""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reporter_id: str = Field(min_length=1, max_length=100)
    merchant_id: str = Field(min_length=1, max_length=100)
    report_type: Literal["fraud", "scam", "service_issue", "technical", "other"]
    rating: Optional[int] = Field(None, ge=1, le=5)
    title: str = Field(min_length=3, max_length=200)
    description: str = Field(min_length=10, max_length=5000)
    transaction_id: Optional[str] = Field(None, max_length=100)
    amount: Optional[float] = Field(None, gt=0)
    metadata: Optional[ReportMetadata] = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "description")
    @classmethod
    def clean_text(cls, v: str) -> str:
        """Clean and normalize text fields"""
        # Strip whitespace
        v = v.strip()
        # Remove multiple spaces
        v = re.sub(r"\s+", " ", v)
        return v

    @field_validator("reporter_id", "merchant_id", "transaction_id")
    @classmethod
    def validate_id_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate ID format - alphanumeric, hyphens, underscores only"""
        if v is None:
            return v
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                f"Invalid ID format: {v}. Only alphanumeric, hyphens, and underscores allowed."
            )
        return v

    @field_validator("description")
    @classmethod
    def check_malicious_content(cls, v: str) -> str:
        # Check for XSS patterns
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
        ]
        for pattern in xss_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potentially malicious HTML/JavaScript detected")

        return v

    def to_dict(self) -> dict:
        """Convert model to dictionary for storage"""
        data = self.model_dump()
        # Convert datetime to string for SQLite
        data["timestamp"] = self.timestamp.isoformat()
        # Convert metadata to JSON string if present
        if self.metadata:
            import json

            data["metadata_json"] = json.dumps(self.metadata.model_dump())
            del data["metadata"]
        else:
            data["metadata_json"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        """Create model from dictionary (from database), tolerant to extra DB columns."""
        import json

        if not isinstance(data, dict):
            raise ValueError("from_dict expects a dict")

        # 1) Keep only fields that belong to the Pydantic model (plus metadata_json helper)
        allowed = set(cls.model_fields.keys()) | {"metadata_json"}
        cleaned = {k: v for k, v in data.items() if k in allowed}

        # 2) Parse timestamp
        if isinstance(cleaned.get("timestamp"), str):
            cleaned["timestamp"] = datetime.fromisoformat(cleaned["timestamp"])

        # 3) Parse metadata_json -> metadata
        if cleaned.get("metadata_json"):
            metadata_dict = json.loads(cleaned["metadata_json"])
            cleaned["metadata"] = ReportMetadata(**metadata_dict)
        cleaned.pop("metadata_json", None)

        return cls(**cleaned)
