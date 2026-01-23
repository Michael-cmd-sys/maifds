"""
maifds_repo/app.py

Unified FastAPI gateway for all Mobile Money Fraud Defense features in this repo.

Key fix:
- Import feature modules via *package paths* so relative imports work.
- Avoid sys.path tricks that cause `model.py` collisions across features.
"""
from __future__ import annotations

import os
import sys
import inspect
import importlib
import asyncio
import dataclasses
from enum import Enum
from datetime import datetime, date

from typing import Optional, Dict, Any, Literal, List

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles # Added for frontend serving
from pydantic import BaseModel, Field, confloat, conint, model_validator
from customer_reputation_system.src.ingestion.report_handler import ReportHandler
from fastapi.middleware.cors import CORSMiddleware




# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path so `mel_dev...` imports work
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()


# -------------------------------------------------------------------------
# Make customer_reputation_system imports like `from src...` work
# -------------------------------------------------------------------------
CRS_ROOT = os.path.join(ROOT, "customer_reputation_system")
if CRS_ROOT not in sys.path:
    sys.path.insert(0, CRS_ROOT)



# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="MAIFDS - (Momo AI Fraud Detection System) MindSpore Powered Fraud Defense System",
    version="0.3.0",
)


# -------------------------------------------------------------------------
# CORS (Vite UI can call the FastAPI backend)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------------------------------------------------

# Serve Static Files (Frontend)
# This allows running everything on port 8000 (Backend + Frontend)
ui_dist_path = os.path.join(ROOT, "ui", "dist")
ui_assets_path = os.path.join(ui_dist_path, "assets")

# Only mount if the assets directory actually exists to prevent RuntimeError
if os.path.exists(ui_dist_path) and os.path.exists(ui_assets_path):
    # Serve assets at /assets
    app.mount("/assets", StaticFiles(directory=ui_assets_path), name="assets")

    # Serve app index
    @app.get("/app/{rest_of_path:path}")
    async def serve_app(rest_of_path: str):
        return FileResponse(os.path.join(ui_dist_path, "index.html"))

    # Serve root index
    @app.get("/")
    async def serve_root():
        return FileResponse(os.path.join(ui_dist_path, "index.html"))

    from fastapi.responses import FileResponse
else:
    print(f"WARNING: UI build not found at {ui_dist_path}. Backend running in API-only mode.")
    print("Run 'npm run build' in ui/ folder to enable backend serving of frontend.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/health")
def v1_health() -> Dict[str, str]:
    return {"status": "ok"}


def _log_global_alert(alert_type: str, severity: str, description: str, entity_id: str = None, entity_name: str = None):
    """Helper to log alerts to the central CRS database for dashboard visibility"""
    try:
        db = _crs_db_manager()
        alert = {
            "alert_type": alert_type,
            "entity_id": entity_id,
            "entity_name": entity_name,
            "risk_score": 1.0 if severity == "critical" else 0.7 if severity == "high" else 0.4,
            "severity": severity,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        db.log_alert(alert)
    except Exception as e:
        print(f"Failed to log global alert: {e}")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _import_module(module_path: str):
    try:
        return importlib.import_module(module_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import '{module_path}': {e}")


def _call_fn_or_class_method(module, names: List[str], *args, **kwargs):
    # 1) direct functions
    for name in names:
        obj = getattr(module, name, None)
        if callable(obj):
            return obj(*args, **kwargs)

    # 2) class instances with methods
    for _, obj in vars(module).items():
        if inspect.isclass(obj):
            try:
                inst = obj()  # allow default ctor only
            except Exception:
                continue
            for name in names:
                meth = getattr(inst, name, None)
                if callable(meth):
                    return meth(*args, **kwargs)

    raise HTTPException(status_code=500, detail=f"No supported function/class-method found. Tried: {names}")


def _safe_jsonable(x: Any) -> Any:
    if x is None:
        return None

    # datetime/date
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    # Enums -> value
    if isinstance(x, Enum):
        return x.value

    # dataclasses (your DataClassification is a dataclass)
    if dataclasses.is_dataclass(x):
        return _safe_jsonable(dataclasses.asdict(x))

    # primitives
    if isinstance(x, (str, int, float, bool)):
        return x

    # lists/tuples
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(i) for i in x]

    # dict
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}

    # Pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return _safe_jsonable(x.model_dump())
        except Exception:
            pass

    # objects that support to_dict()
    if hasattr(x, "to_dict"):
        try:
            return _safe_jsonable(x.to_dict())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(x, "dict"):
        try:
            return _safe_jsonable(x.dict())
        except Exception:
            pass

    # fallback
    return str(x)



# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------

# mel_dev call_triggered_defense feature request models
class CallTriggeredDefenseRequest(BaseModel):
    tx_amount: confloat(ge=0) = Field(..., description="Transaction amount (e.g., GHS)")
    recipient_first_time: conint(ge=0, le=1) = Field(..., description="1 if first time sending to recipient else 0")
    call_to_tx_delta_seconds: confloat(ge=0) = Field(..., description="Seconds between call and transaction")
    contact_list_flag: conint(ge=0, le=1) = Field(..., description="1 if recipient is in contacts else 0")
    nlp_suspicion_score: confloat(ge=0.0, le=1.0) = Field(..., description="0.0–1.0 suspicion score")

# mel_dev click_tx_link_correlation feature request models
class ClickTxCorrelationRequest(BaseModel):
    tx_amount: Optional[confloat(ge=0)] = Field(None, description="Transaction amount")
    amount: Optional[confloat(ge=0)] = Field(None, description="Alias for tx_amount")

    url_reported_flag: conint(ge=0, le=1)
    time_between_click_and_tx: confloat(ge=0)

    url_risk_score: Optional[confloat(ge=0.0, le=1.0)] = 0.0
    clicked_recently: Optional[conint(ge=0, le=1)] = 0
    device_click_count_1d: Optional[conint(ge=0)] = 0
    user_click_count_1d: Optional[conint(ge=0)] = 0

    time_since_last_tx_seconds: Optional[confloat(ge=0)] = 0.0
    tx_hour: Optional[conint(ge=0, le=23)] = 0
    tx_dayofweek: Optional[conint(ge=0, le=6)] = 0
    url_hash_numeric: Optional[confloat(ge=0)] = 0.0

    @model_validator(mode="after")
    def _ensure_amount(self):
        if self.tx_amount is None and self.amount is None:
            raise ValueError("Either tx_amount or amount must be provided")
        if self.tx_amount is None:
            self.tx_amount = self.amount
        return self


# mel_dev proactive_pre_tx_warning feature request models
class ProactiveWarningRequest(BaseModel):
    recent_risky_clicks_7d: conint(ge=0) = 0
    recent_scam_calls_7d: conint(ge=0) = 0
    scam_campaign_intensity: confloat(ge=0.0, le=1.0) = 0.0
    device_age_days: confloat(ge=0.0) = 180.0
    is_new_device: conint(ge=0, le=1) = 0
    tx_count_total: conint(ge=0) = 0
    avg_tx_amount: confloat(ge=0.0) = 0.0
    max_tx_amount: confloat(ge=0.0) = 0.0
    historical_fraud_flag: conint(ge=0, le=1) = 0
    is_in_campaign_cohort: conint(ge=0, le=1) = 0
    user_risk_score: confloat(ge=0.0, le=1.0) = 0.0

# mel_dev telco_notification_webhook feature request models
class TelcoNotifyRequest(BaseModel):
    incident_id: str = Field(..., min_length=1)
    suspected_number: str = Field(..., min_length=1)
    affected_accounts: List[str] = Field(default_factory=list)
    observed_evidence: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = Field(..., min_length=1)

# mel_dev user_sms_alert feature request models
class UserSmsAlertRequest(BaseModel):
    phone: str = Field(..., min_length=5, description="Recipient phone (055... or +233...)")
    threat_type: str = Field("GENERAL", min_length=3)
    suspected_number: Optional[str] = None
    message: Optional[str] = None  # if provided, overrides template
    ref: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)

# mel_dev orchestrator request model
class OrchestratorRequest(BaseModel):
    user_phone: Optional[str] = None
    suspected_number: Optional[str] = None
    incident_id: Optional[str] = None
    affected_accounts: List[str] = Field(default_factory=list)
    timestamps: Dict[str, Any] = Field(default_factory=dict)
    ref: Optional[str] = None

    call_triggered_defense: Optional[Dict[str, Any]] = None
    click_tx_correlation: Optional[Dict[str, Any]] = None
    proactive_pre_tx_warning: Optional[Dict[str, Any]] = None

# maifds_services Blacklist_Watchlist_Service request models
class BlacklistAddRequest(BaseModel):
    value: str = Field(..., min_length=1)
    list_type: str = Field(..., min_length=1)
    reason: str = "manual_add"
    severity: str = "high"
    source: str = "api"
    additional_info: Dict[str, Any] = Field(default_factory=dict)

# maifds_services Blacklist_Watchlist_Service request models
class BlacklistRemoveRequest(BaseModel):
    value: str = Field(..., min_length=1)
    list_type: str = Field(..., min_length=1)

# maifds_services Blacklist_Watchlist_Service request models
class BlacklistCheckRequest(BaseModel):
    phone_number: Optional[str] = None
    device_id: Optional[str] = None
    url: Optional[str] = None

BlacklistCheckRequest.model_rebuild()


# maifds_services Phishing_Ad_Referral_Channel_Detector request models
class PhishingScoreRequest(BaseModel):
    url: str = Field(..., min_length=1)

class DomainIntelRequest(BaseModel):
    domain: str = Field(..., min_length=1, description="Domain name to analyze")

# maifds_services CRS feature request models
class CRSReportSubmitRequest(BaseModel):
    reporter_id: str
    merchant_id: str
    report_type: Literal["fraud", "scam", "service_issue", "technical", "other"]
    rating: Optional[int] = None
    title: str
    description: str
    transaction_id: Optional[str] = None
    amount: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class CRSUpdateRiskRequest(BaseModel):
    entity_type: str  # "agent" | "mule"
    entity_id: str


# 🔧 REQUIRED for Pydantic v2 + FastAPI
CRSReportSubmitRequest.model_rebuild()
CRSUpdateRiskRequest.model_rebuild()


# -------------------------
# maifds_governance models
# -------------------------

class GovernanceAuditEventRequest(BaseModel):
    payload: Dict[str, Any]
    method: str = "process_event"   # exact AuditProcessor method to call

class PrivacyClassifyRequest(BaseModel):
    text: str
    method: str = "classify_data"   # ✅ FIXED default

class PrivacyAnonymizeRequest(BaseModel):
    text: str
    strategy: Literal["mask", "hash", "redact"] = "mask"
    method: str = "anonymize_field"  # ✅ FIXED default (exists in DataAnonymizer)

# -----------------------------------------------------------------------------
# maifds_governance: privacy service request models
# -----------------------------------------------------------------------------

class PrivacyBatchClassifyRequest(BaseModel):
    items: List[str] = Field(..., min_length=1)


class PrivacyClassifyDataRequest(BaseModel):
    data: Any
    data_id: Optional[str] = None


class PrivacyClassifyRecordRequest(BaseModel):
    record: Dict[str, Any]
    record_id: Optional[str] = None


class PrivacyAnonymizeRecordRequest(BaseModel):
    record: Dict[str, Any]
    field_mapping: Optional[Dict[str, str]] = None


class PrivacyTokenRequest(BaseModel):
    token: str


class ConsentCreateRequest(BaseModel):
    user_id: str
    template_id: str
    metadata: Optional[Dict[str, Any]] = None


class ConsentActionRequest(BaseModel):
    consent_id: str
    user_id: str
    reason: Optional[str] = None


class AccessCheckRequest(BaseModel):
    user_id: str
    role: str
    resource: str
    operation: str
    purpose: str
    sensitivity: str
    context: Optional[Dict[str, Any]] = None


class AccessRequestCreate(BaseModel):
    user_id: str
    role: str
    resource: str
    operation: str
    purpose: str
    sensitivity: str
    justification: str
    expires_hours: Optional[int] = None


class AccessApprovalRequest(BaseModel):
    request_id: str
    approver_id: str


class AccessDenyRequest(BaseModel):
    request_id: str
    approver_id: str
    reason: str


class GDPRCreateRequest(BaseModel):
    user_id: str
    right: str
    details: Optional[str] = ""


class GDPRProcessAccessRequest(BaseModel):
    request_id: str
    user_data: Dict[str, Any]


class GDPRPortabilityRequest(BaseModel):
    request_id: str
    user_data: Dict[str, Any]


class GDPRBreachReportRequest(BaseModel):
    severity: str
    affected_users: int
    data_types: List[str]
    description: str


# -----------------------------------------------------------------------------
# maifds_governance: privacy service singletons
# -----------------------------------------------------------------------------
from typing import Tuple

_PRIVACY_CLASSIFIER = None
_PRIVACY_ANONYMIZER = None
_PRIVACY_CONSENT = None
_PRIVACY_ACCESS = None
_PRIVACY_GDPR = None


def _privacy_services():
    """
    Build/reuse privacy service singletons.
    Keeps in-memory state (consent history, access logs, token maps, etc.)
    """
    global _PRIVACY_CLASSIFIER, _PRIVACY_ANONYMIZER, _PRIVACY_CONSENT, _PRIVACY_ACCESS, _PRIVACY_GDPR

    if _PRIVACY_CLASSIFIER is None:
        from maifds_governance.privacy.data_classifier import DataClassifier
        _PRIVACY_CLASSIFIER = DataClassifier()

    if _PRIVACY_ANONYMIZER is None:
        from maifds_governance.privacy.anonymizer import DataAnonymizer
        _PRIVACY_ANONYMIZER = DataAnonymizer()

    if _PRIVACY_CONSENT is None:
        from maifds_governance.privacy.consent_manager import ConsentManager
        _PRIVACY_CONSENT = ConsentManager()

    if _PRIVACY_ACCESS is None:
        from maifds_governance.privacy.access_control import PrivacyAccessController
        _PRIVACY_ACCESS = PrivacyAccessController()

    if _PRIVACY_GDPR is None:
        from maifds_governance.privacy.gdpr_compliance import GDPRComplianceManager
        _PRIVACY_GDPR = GDPRComplianceManager()

    return {
        "classifier": _PRIVACY_CLASSIFIER,
        "anonymizer": _PRIVACY_ANONYMIZER,
        "consent": _PRIVACY_CONSENT,
        "access": _PRIVACY_ACCESS,
        "gdpr": _PRIVACY_GDPR,
    }




# -----------------------------------------------------------------------------
# mel_dev endpoints (5 features here)
# -----------------------------------------------------------------------------
@app.post("/v1/call-triggered-defense")
def call_triggered_defense(
    req: CallTriggeredDefenseRequest,
    debug: bool = Query(False, description="If true, returns debug info about payload passed into the engine.")
) -> Dict[str, Any]:
    mod = _import_module("mel_dev.features.call_triggered_defense.src.inference")

    payload = req.model_dump()

    try:
        result = _call_fn_or_class_method(
            mod,
            ["run_inference", "predict", "infer", "score"],
            payload,
            debug=debug
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"call_triggered_defense failed: {e}")

    out = {"feature": "call_triggered_defense", "result": _safe_jsonable(result)}
    if debug:
        out["debug"] = {"payload_sent_to_engine": payload}
    return out


@app.post("/v1/click-tx-correlation")
def click_tx_correlation(
    req: ClickTxCorrelationRequest,
    debug: bool = Query(False, description="If true, returns debug info about payload passed into the engine.")
) -> Dict[str, Any]:
    mod = _import_module("mel_dev.features.click_tx_link_correlation.src.inference")

    payload = req.model_dump()

    # map tx_amount -> amount for feature compatibility (optional but helpful)
    payload["amount"] = payload.get("tx_amount", payload.get("amount", 0.0))

    try:
        result = _call_fn_or_class_method(
            mod,
            ["run_inference", "predict", "infer", "score"],
            payload,
            debug=debug
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"click_tx_correlation failed: {e}")

    out = {"feature": "click_tx_link_correlation", "result": _safe_jsonable(result)}
    if debug:
        out["debug"] = {"payload_sent_to_engine": payload}
    return out

@app.post("/v1/proactive-pre-tx-warning")
def proactive_pre_tx_warning(req: ProactiveWarningRequest, debug: bool = Query(False)):
    mod = _import_module("mel_dev.features.proactive_pre_tx_warning.src.inference")
    payload = req.model_dump()
    result = _call_fn_or_class_method(mod, ["run_inference"], payload, debug=debug)
    out = {"feature": "proactive_pre_tx_warning", "result": _safe_jsonable(result)}
    if debug:
        out["debug"] = {"payload_sent_to_engine": payload}
    return out


@app.post("/v1/telco-notify")
def telco_notify(req: TelcoNotifyRequest) -> Dict[str, Any]:
    client_mod = _import_module("mel_dev.features.telco_notification_webhook.src.client")

    incident_obj = req.model_dump()

    try:
        schemas_mod = _import_module("mel_dev.features.telco_notification_webhook.src.schemas")
        TelcoIncidentPayload = getattr(schemas_mod, "TelcoIncidentPayload", None)
        if TelcoIncidentPayload is not None:
            incident_obj = TelcoIncidentPayload(**incident_obj)  # <-- force schema object
    except Exception as e:
        # keep dict fallback
        pass

    client_cls = getattr(client_mod, "TelcoNotificationClient", None)
    if client_cls is not None:
        client = client_cls()
        if hasattr(client, "send_incident"):
            resp = client.send_incident(incident_obj)
            return {"feature": "telco_notification_webhook", "result": _safe_jsonable(resp)}

    resp = _call_fn_or_class_method(client_mod, ["send_incident", "notify_telco", "push_incident"], incident_obj)
    return {"feature": "telco_notification_webhook", "result": _safe_jsonable(resp)}


@app.post("/v1/user-sms-alert")
def user_sms_alert(req: UserSmsAlertRequest) -> Dict[str, Any]:
    from mel_dev.features.user_sms_alert.src.phone import normalize_gh_number
    from mel_dev.features.user_sms_alert.src.client import MoolreSmsClient
    from mel_dev.features.user_sms_alert.src.message_builder import build_user_sms

    # normalize to digits-only 233...
    recipient = normalize_gh_number(req.phone)

    # build message
    ctx = dict(req.context or {})
    if req.suspected_number:
        ctx["suspected_number"] = req.suspected_number

    sms_text = req.message or build_user_sms(req.threat_type, ctx)

    client = MoolreSmsClient()
    result = client.send_sms(recipient=recipient, message=sms_text, ref=req.ref)

    return {
        "feature": "user_sms_alert",
        "result": _safe_jsonable({
            "recipient_normalized": recipient,
            "sms_text": sms_text,
            "provider": result.to_dict(),
        })
    }


@app.post("/v1/orchestrate")
def orchestrate(req: OrchestratorRequest, debug: bool = Query(False)):
    from mel_dev.features.orchestrator.src.orchestrator import run_orchestrator
    payload = req.model_dump()
    return {"feature": "orchestrator", "result": _safe_jsonable(run_orchestrator(payload, debug=debug))}




# -----------------------------------------------------------------------------
# maifds_services endpoints (ignore Proactive_Warning_Service)
# -----------------------------------------------------------------------------
@app.post("/v1/blacklist/check")
def blacklist_check(req: BlacklistCheckRequest = Body(...)) -> Dict[str, Any]:
    mod = _import_module("maifds_services.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")

    signals = req.model_dump(exclude_none=True)
    if not signals:
        raise HTTPException(status_code=400, detail="Provide at least one of: phone_number, device_id, url")

    # call the service singleton directly
    svc = getattr(mod, "_get_service")()
    result = svc.check_blacklist(signals)

    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}

@app.post("/v1/blacklist/add")
def blacklist_add(req: BlacklistAddRequest) -> Dict[str, Any]:
    mod = _import_module("maifds_services.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")
    metadata = {
        "reason": req.reason,
        "severity": req.severity,
        "source": req.source,
        "additional_info": req.additional_info,
    }
    result = _call_fn_or_class_method(mod, ["add_value"], req.value, req.list_type, metadata)
    # persist bloom filters (optional but good for demo)
    svc = getattr(mod, "_get_service")()
    svc.save_bloom_filters()
    
    # Log to dashboard alerts
    _log_global_alert(
        alert_type="blacklist_add",
        severity=req.severity,
        description=f"Manual addition to {req.list_type} blacklist: {req.value}. Reason: {req.reason}",
        entity_id=req.value,
        entity_name="Blacklist Entry"
    )
    
    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}

@app.post("/v1/blacklist/remove")
def blacklist_remove(req: BlacklistRemoveRequest) -> Dict[str, Any]:
    mod = _import_module("maifds_services.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")
    result = _call_fn_or_class_method(mod, ["remove_value"], req.value, req.list_type)
    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}

@app.get("/v1/blacklist/stats")
def blacklist_stats() -> Dict[str, Any]:
    mod = _import_module("maifds_services.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")
    result = _call_fn_or_class_method(mod, ["stats"])
    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}

@app.post("/v1/blacklist/rebuild")
def blacklist_rebuild() -> Dict[str, Any]:
    mod = _import_module("maifds_services.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")
    result = _call_fn_or_class_method(mod, ["rebuild"])
    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}


@app.post("/v1/phishing-ad-referral/score")
def phishing_ad_referral_score(req: PhishingScoreRequest) -> Dict[str, Any]:
    mod = _import_module("maifds_services.Phishing_Ad_Referral_Channel_Detector.src.mindspore_detector")
    #mindspore_detector expects referrer_url
    signals = {"referrer_url": req.url}
    result = _call_fn_or_class_method(mod, ["detect_phishing"], signals)
    return {"feature": "phishing_ad_referral_channel_detector", "result": _safe_jsonable(result)}

@app.post("/v1/phishing-ad-referral/domain-intel")
def domain_intelligence(req: DomainIntelRequest) -> Dict[str, Any]:
    from maifds_services.Phishing_Ad_Referral_Channel_Detector.src.domain_intelligence import DomainIntelligence
    intel = DomainIntelligence()
    result = intel.analyze_domain(req.domain)
    return {"feature": "domain_intelligence", "result": _safe_jsonable(result)}


# -----------------------------------------------------------------------------
# customer_reputation_system endpoints
# -----------------------------------------------------------------------------
def _crs_db_manager():
    from customer_reputation_system.src.storage.database import DatabaseManager
    db_path = os.path.join(ROOT, "customer_reputation_system", "data", "database", "reports.db")
    return DatabaseManager(db_path)


def _crs_risk_api():
    from customer_reputation_system.src.api.realtime_risk_api import RealTimeRiskAPI
    return RealTimeRiskAPI(_crs_db_manager())



@app.post("/v1/customer-reputation/report/submit")
def crs_submit_report(req: CRSReportSubmitRequest) -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    payload = req.model_dump(exclude_none=True)
    result = handler.submit_report(payload)
    
    # Log significant reports as alerts
    if req.report_type in ["fraud", "scam"]:
        _log_global_alert(
            alert_type="reputation_report",
            severity="high",
            description=f"New {req.report_type} report against {req.merchant_id}: {req.title}",
            entity_id=req.merchant_id,
            entity_name="Suspect Entity"
        )
        
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.get("/v1/customer-reputation/report/{report_id}")
def crs_get_report(report_id: str) -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    report = handler.get_report(report_id)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(report)}


@app.get("/v1/customer-reputation/stats")
def crs_stats() -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    stats = handler.get_statistics()
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(stats)}


@app.get("/v1/customer-reputation/merchant/{merchant_id}/reports")
def crs_merchant_reports(merchant_id: str) -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    reports = handler.get_merchant_reports(merchant_id)
    return {
        "feature": "customer_reputation_system",
        "result": _safe_jsonable({"merchant_id": merchant_id, "reports": reports}),
    }


@app.get("/v1/customer-reputation/reporter/{reporter_id}/reports")
def crs_reporter_reports(reporter_id: str) -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    reports = handler.get_reporter_reports(reporter_id)
    return {
        "feature": "customer_reputation_system",
        "result": _safe_jsonable({"reporter_id": reporter_id, "reports": reports}),
    }


@app.get("/v1/customer-reputation/reports/recent")
def crs_recent_reports(limit: int = 10) -> Dict[str, Any]:
    from customer_reputation_system.src.ingestion.report_handler import ReportHandler
    handler = ReportHandler(db_manager=_crs_db_manager())

    reports = handler.get_recent_reports(limit)
    return {
        "feature": "customer_reputation_system",
        "result": _safe_jsonable(reports),
    }


@app.get("/v1/customer-reputation/agent/{agent_id}/risk")
def crs_agent_risk(agent_id: str) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.get_agent_risk_score(agent_id)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.get("/v1/customer-reputation/merchant/{merchant_id}/risk")
def crs_merchant_risk(merchant_id: str) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.get_merchant_risk_assessment(merchant_id)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.get("/v1/customer-reputation/transactions/suspicious")
def crs_suspicious_tx(hours: int = Query(24, ge=1, le=720)) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.detect_suspicious_transactions(time_window_hours=hours)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.get("/v1/customer-reputation/network/overview")
def crs_network_overview(network_id: Optional[str] = None) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.get_network_risk_overview(network_id=network_id)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.get("/v1/customer-reputation/alerts")
def crs_alerts(threshold: float = Query(0.8, ge=0.0, le=1.0)) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.get_risk_alerts(severity_threshold=threshold)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}


@app.post("/v1/customer-reputation/risk/update")
def crs_update_risk(req: CRSUpdateRiskRequest) -> Dict[str, Any]:
    api = _crs_risk_api()
    result = api.update_risk_scores_realtime(req.entity_type, req.entity_id)
    return {"feature": "customer_reputation_system", "result": _safe_jsonable(result)}



# -----------------------------------------------------------------------------
# maifds_governance: audit_service endpoints
# -----------------------------------------------------------------------------

@app.post("/v1/governance/audit/event")
async def governance_audit_event(req: GovernanceAuditEventRequest):
    mod = importlib.import_module("maifds_governance.audit_service")

    method = req.method
    payload = req.payload or {}

    # Prefer the async ingest if it exists (you created it)
    if hasattr(mod, "ingest_event_async"):
        return {
            "feature": "governance_audit_service",
            "result": await mod.ingest_event_async(payload, method=method),
        }

    # Fallback to sync ingest (if async not present)
    if hasattr(mod, "ingest_event"):
        return {
            "feature": "governance_audit_service",
            "result": mod.ingest_event(payload, method=method),
        }

    # If neither exists, show what IS available
    public = [x for x in dir(mod) if not x.startswith("_")]
    return {
        "feature": "governance_audit_service",
        "result": {"error": "No ingest function found", "available": public},
    }

@app.get("/v1/governance/audit/health")
def governance_audit_health() -> Dict[str, Any]:
    mod = _import_module("maifds_governance.audit_service")
    result = _call_fn_or_class_method(mod, ["get_audit_health"])
    return {"feature": "governance_audit_service", "result": _safe_jsonable(result)}


@app.get("/v1/governance/audit/stats")
def governance_audit_stats(limit: int = Query(20, ge=0, le=200)) -> Dict[str, Any]:
    mod = _import_module("maifds_governance.audit_service")
    result = _call_fn_or_class_method(mod, ["get_audit_stats"], limit)
    return {"feature": "governance_audit_service", "result": _safe_jsonable(result)}

@app.on_event("startup")
async def start_governance_audit_background_tasks():
    from maifds_governance.audit_service import _get_services

    s = _get_services()
    bus = s["event_bus"]
    ap = s["audit_processor"]

    # Start EventBus loop
    if not getattr(bus, "_processing", False):
        asyncio.create_task(bus.start_processing())

    # Start AuditProcessor loop
    if not getattr(ap, "processing", False):
        asyncio.create_task(ap.start_processing())



# -----------------------------------------------------------------------------
# maifds_governance: privacy endpoints
# -----------------------------------------------------------------------------

@app.post("/v1/governance/privacy/classify")
def governance_privacy_classify(req: PrivacyClassifyRequest) -> Dict[str, Any]:
    mod = _import_module("maifds_governance.privacy")

    # IMPORTANT: DataClassifier has classify_data(), not classify()
    method = req.method or "classify_data"

    result = _call_fn_or_class_method(mod, ["classify_text"], req.text, method=method)
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.post("/v1/governance/privacy/anonymize")
def governance_privacy_anonymize(req: PrivacyAnonymizeRequest) -> Dict[str, Any]:
    mod = _import_module("maifds_governance.privacy")
    result = _call_fn_or_class_method(mod, ["anonymize_text"], req.text, req.strategy, method=req.method)
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}

@app.get("/v1/governance/privacy/health")
def governance_privacy_health() -> Dict[str, Any]:
    mod = _import_module("maifds_governance.privacy")
    result = _call_fn_or_class_method(mod, ["health"])
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


# -----------------------------------------------------------------------------
# maifds_governance: privacy extended endpoints
# -----------------------------------------------------------------------------

@app.get("/v1/governance/privacy/pii-types")
def governance_privacy_pii_types() -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]
    result = classifier.detect_pii_types("dummy test@example.com +233554123456 197.0.0.1")
    return {"feature": "governance_privacy", "result": _safe_jsonable({"pii_types_supported": list(result.keys()) if isinstance(result, dict) else result})}


@app.post("/v1/governance/privacy/classify/batch")
def governance_privacy_classify_batch(req: PrivacyBatchClassifyRequest) -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]

    results = []
    for i, text in enumerate(req.items):
        results.append(_safe_jsonable(classifier.classify_data(text, data_id=f"batch_{i}")))
    return {"feature": "governance_privacy", "result": {"count": len(results), "items": results}}


@app.post("/v1/governance/privacy/classify/data")
def governance_privacy_classify_data(req: PrivacyClassifyDataRequest) -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]
    result = classifier.classify_data(req.data, data_id=req.data_id)
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.post("/v1/governance/privacy/classify/record")
def governance_privacy_classify_record(req: PrivacyClassifyRecordRequest) -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]
    result = classifier.classify_record(req.record, record_id=req.record_id)  # exists per your health methods
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.get("/v1/governance/privacy/classifications/stats")
def governance_privacy_classification_stats() -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]
    result = classifier.get_classification_statistics()
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.get("/v1/governance/privacy/classifications/summary")
def governance_privacy_data_summary() -> Dict[str, Any]:
    s = _privacy_services()
    classifier = s["classifier"]
    result = classifier.get_data_summary()
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


# -------------------------
# Anonymization routes
# -------------------------

@app.post("/v1/governance/privacy/anonymize/record")
def governance_privacy_anonymize_record(req: PrivacyAnonymizeRecordRequest) -> Dict[str, Any]:
    s = _privacy_services()
    anonymizer = s["anonymizer"]
    result = anonymizer.anonymize_record(req.record, field_mapping=req.field_mapping)
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.post("/v1/governance/privacy/anonymize/summary")
def governance_privacy_anonymize_summary(req: PrivacyAnonymizeRecordRequest) -> Dict[str, Any]:
    s = _privacy_services()
    anonymizer = s["anonymizer"]
    anon = anonymizer.anonymize_record(req.record, field_mapping=req.field_mapping)
    result = anonymizer.get_anonymization_summary(req.record, anon)
    return {"feature": "governance_privacy", "result": _safe_jsonable(result)}


@app.post("/v1/governance/privacy/anonymize/deanonymize")
def governance_privacy_deanonymize(req: PrivacyTokenRequest) -> Dict[str, Any]:
    s = _privacy_services()
    anonymizer = s["anonymizer"]
    result = anonymizer.deanonymize(req.token)
    return {"feature": "governance_privacy", "result": _safe_jsonable({"token": req.token, "value": result})}


# -------------------------
# Consent routes
# -------------------------

@app.post("/v1/governance/privacy/consent/request")
def governance_privacy_consent_request(req: ConsentCreateRequest) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    consent_id = cm.create_consent_request(req.user_id, req.template_id, metadata=req.metadata)
    return {"feature": "governance_privacy", "result": {"status": "ok", "consent_id": consent_id}}


@app.post("/v1/governance/privacy/consent/grant")
def governance_privacy_consent_grant(req: ConsentActionRequest) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    ok = cm.grant_consent(req.consent_id, req.user_id)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed", "granted": ok}}


@app.post("/v1/governance/privacy/consent/deny")
def governance_privacy_consent_deny(req: ConsentActionRequest) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    ok = cm.deny_consent(req.consent_id, req.user_id)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed", "denied": ok}}


@app.post("/v1/governance/privacy/consent/withdraw")
def governance_privacy_consent_withdraw(req: ConsentActionRequest) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    ok = cm.withdraw_consent(req.consent_id, req.user_id, reason=req.reason)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed", "withdrawn": ok}}


@app.get("/v1/governance/privacy/consent/{user_id}")
def governance_privacy_get_user_consents(user_id: str) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    consents = cm.get_user_consents(user_id)
    return {"feature": "governance_privacy", "result": _safe_jsonable({"user_id": user_id, "consents": consents})}


@app.get("/v1/governance/privacy/consent/{user_id}/summary")
def governance_privacy_get_user_consent_summary(user_id: str) -> Dict[str, Any]:
    s = _privacy_services()
    cm = s["consent"]
    summary = cm.get_consent_summary(user_id)
    return {"feature": "governance_privacy", "result": _safe_jsonable(summary)}


# -------------------------
# Access control routes
# -------------------------

def _enum_or_400(enum_cls, value: str):
    try:
        return enum_cls(value)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid {enum_cls.__name__}: {value}. Allowed: {[e.value for e in enum_cls]}")


@app.post("/v1/governance/privacy/access/check")
def governance_privacy_access_check(req: AccessCheckRequest) -> Dict[str, Any]:
    from fastapi import HTTPException
    from maifds_governance.privacy.access_control import UserRole, AccessPurpose, DataSensitivity

    s = _privacy_services()
    ac = s["access"]

    role = _enum_or_400(UserRole, req.role)
    purpose = _enum_or_400(AccessPurpose, req.purpose)
    sensitivity = _enum_or_400(DataSensitivity, req.sensitivity)

    granted, reason = ac.check_access(
        user_id=req.user_id,
        role=role,
        resource=req.resource,
        operation=req.operation,
        purpose=purpose,
        sensitivity=sensitivity,
        context=req.context or {},
    )
    return {"feature": "governance_privacy", "result": {"granted": granted, "reason": reason}}


@app.post("/v1/governance/privacy/access/request")
def governance_privacy_access_request(req: AccessRequestCreate) -> Dict[str, Any]:
    from fastapi import HTTPException
    from maifds_governance.privacy.access_control import UserRole, AccessPurpose, DataSensitivity

    s = _privacy_services()
    ac = s["access"]

    role = _enum_or_400(UserRole, req.role)
    purpose = _enum_or_400(AccessPurpose, req.purpose)
    sensitivity = _enum_or_400(DataSensitivity, req.sensitivity)

    request_id = ac.request_access(
        user_id=req.user_id,
        role=role,
        resource=req.resource,
        operation=req.operation,
        purpose=purpose,
        sensitivity=sensitivity,
        justification=req.justification,
        expires_hours=req.expires_hours,
    )
    return {"feature": "governance_privacy", "result": {"status": "ok", "request_id": request_id}}


@app.post("/v1/governance/privacy/access/approve")
def governance_privacy_access_approve(req: AccessApprovalRequest) -> Dict[str, Any]:
    s = _privacy_services()
    ac = s["access"]
    ok = ac.approve_request(req.request_id, req.approver_id)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed", "approved": ok}}


@app.post("/v1/governance/privacy/access/deny")
def governance_privacy_access_deny(req: AccessDenyRequest) -> Dict[str, Any]:
    s = _privacy_services()
    ac = s["access"]
    ok = ac.deny_request(req.request_id, req.approver_id, req.reason)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed", "denied": ok}}


@app.get("/v1/governance/privacy/access/logs")
def governance_privacy_access_logs(user_id: Optional[str] = None) -> Dict[str, Any]:
    s = _privacy_services()
    ac = s["access"]
    logs = ac.get_access_logs(user_id=user_id)
    return {"feature": "governance_privacy", "result": _safe_jsonable({"user_id": user_id, "logs": logs})}


@app.get("/v1/governance/privacy/access/stats")
def governance_privacy_access_stats() -> Dict[str, Any]:
    s = _privacy_services()
    ac = s["access"]
    stats = ac.get_access_statistics()
    return {"feature": "governance_privacy", "result": _safe_jsonable(stats)}


# -------------------------
# GDPR routes
# -------------------------

@app.post("/v1/governance/privacy/gdpr/request")
def governance_privacy_gdpr_request(req: GDPRCreateRequest) -> Dict[str, Any]:
    from fastapi import HTTPException
    from maifds_governance.privacy.gdpr_compliance import GDPRRight

    s = _privacy_services()
    gdpr = s["gdpr"]

    right = _enum_or_400(GDPRRight, req.right)
    request_id = gdpr.create_data_subject_request(req.user_id, right, details=req.details or "")
    return {"feature": "governance_privacy", "result": {"status": "ok", "request_id": request_id}}


@app.post("/v1/governance/privacy/gdpr/process/access")
def governance_privacy_gdpr_process_access(req: GDPRProcessAccessRequest) -> Dict[str, Any]:
    s = _privacy_services()
    gdpr = s["gdpr"]
    ok = gdpr.process_access_request(req.request_id, req.user_data)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed"}}


@app.post("/v1/governance/privacy/gdpr/process/portability")
def governance_privacy_gdpr_process_portability(req: GDPRPortabilityRequest) -> Dict[str, Any]:
    s = _privacy_services()
    gdpr = s["gdpr"]
    ok = gdpr.process_portability_request(req.request_id, req.user_data)
    return {"feature": "governance_privacy", "result": {"status": "ok" if ok else "failed"}}


@app.post("/v1/governance/privacy/gdpr/breach/report")
def governance_privacy_gdpr_breach_report(req: GDPRBreachReportRequest) -> Dict[str, Any]:
    from fastapi import HTTPException
    from maifds_governance.privacy.gdpr_compliance import BreachSeverity

    s = _privacy_services()
    gdpr = s["gdpr"]

    severity = _enum_or_400(BreachSeverity, req.severity)
    breach_id = gdpr.report_data_breach(severity, req.affected_users, req.data_types, req.description)
    return {"feature": "governance_privacy", "result": {"status": "ok", "breach_id": breach_id}}


@app.get("/v1/governance/privacy/gdpr/report")
def governance_privacy_gdpr_report() -> Dict[str, Any]:
    s = _privacy_services()
    gdpr = s["gdpr"]
    report = gdpr.generate_compliance_report()
    return {"feature": "governance_privacy", "result": _safe_jsonable(report)}
