from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


def utc_iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class IncidentEvidence:
    """
    Minimal representation of evidence we send to telco, without PII.
    """
    cdr_ids: List[str]            # call detail record IDs or hashes
    click_hashes: List[str]       # URL hash IDs or tokens
    model_scores: Dict[str, float] # feature_name -> fraud probability
    rule_flags: Dict[str, bool]    # feature_name -> rule triggered


@dataclass
class TelcoIncidentPayload:
    """
    Structured payload we POST to the telco webhook.
    """
    incident_id: str
    created_at: str
    suspected_number: Optional[str]
    affected_accounts: List[str]
    recommended_action: str              # e.g. "TEMP_BLOCK_NUMBER", "INVESTIGATE_CAMPAIGN"
    severity: str                        # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    source_system: str                   # e.g. "mindspore_fraud_platform"
    environment: str                     # "dev", "staging", "prod"
    evidence: IncidentEvidence
    metadata: Dict[str, Any]

    @staticmethod
    def new(
        suspected_number: Optional[str],
        affected_accounts: List[str],
        recommended_action: str,
        severity: str,
        source_system: str,
        environment: str,
        evidence: IncidentEvidence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TelcoIncidentPayload":
        return TelcoIncidentPayload(
            incident_id=str(uuid.uuid4()),
            created_at=utc_iso_now(),
            suspected_number=suspected_number,
            affected_accounts=affected_accounts,
            recommended_action=recommended_action,
            severity=severity,
            source_system=source_system,
            environment=environment,
            evidence=evidence,
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = asdict(self.evidence)
        return payload


@dataclass
class TelcoWebhookResponse:
    """
    Basic representation of the telco side response.
    We store this for audit + model retraining later.
    """
    status_code: int
    success: bool
    telco_reference_id: Optional[str]
    raw_body: Optional[str]
    received_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
