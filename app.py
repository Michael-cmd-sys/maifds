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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, confloat, conint, model_validator
from typing import Optional
from pydantic import confloat, conint


# -----------------------------------------------------------------------------
# Ensure repo root is on sys.path so `mel_dev...` imports work
# -----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="MAIFDS - MindSpore Powered Fraud Defense System",
    version="0.3.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/health")
def v1_health() -> Dict[str, str]:
    return {"status": "ok"}


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
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "to_dict"):
        try:
            return x.to_dict()
        except Exception:
            pass
    if hasattr(x, "dict"):
        try:
            return x.dict()
        except Exception:
            pass
    return str(x)


# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------
class CallTriggeredDefenseRequest(BaseModel):
    tx_amount: confloat(ge=0) = Field(..., description="Transaction amount (e.g., GHS)")
    recipient_first_time: conint(ge=0, le=1) = Field(..., description="1 if first time sending to recipient else 0")
    call_to_tx_delta_seconds: confloat(ge=0) = Field(..., description="Seconds between call and transaction")
    contact_list_flag: conint(ge=0, le=1) = Field(..., description="1 if recipient is in contacts else 0")
    nlp_suspicion_score: confloat(ge=0.0, le=1.0) = Field(..., description="0.0–1.0 suspicion score")


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



class ProactiveWarningRequest(BaseModel):
    user_id: Optional[str] = None
    tx_amount: confloat(ge=0) = 0
    recently_clicked: conint(ge=0, le=1) = 0
    recently_called: conint(ge=0, le=1) = 0
    active_campaign_indicator: conint(ge=0, le=1) = 0
    new_device_flag: conint(ge=0, le=1) = 0


class TelcoNotifyRequest(BaseModel):
    incident_id: str = Field(..., min_length=1)
    suspected_number: str = Field(..., min_length=1)
    affected_accounts: List[str] = Field(default_factory=list)
    observed_evidence: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = Field(..., min_length=1)


class BlacklistCheckRequest(BaseModel):
    value: str = Field(..., min_length=1)
    list_type: str = Field(..., min_length=1)


class PhishingScoreRequest(BaseModel):
    url: str = Field(..., min_length=1)


# -----------------------------------------------------------------------------
# mel_dev endpoints (your 4)
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
def proactive_pre_tx_warning(req: ProactiveWarningRequest) -> Dict[str, Any]:
    mod = _import_module("mel_dev.features.proactive_pre_tx_warning.src.inference")
    payload = req.model_dump()
    result = _call_fn_or_class_method(mod, ["run_inference", "predict", "infer", "score", "select_cohort"], payload)
    return {"feature": "proactive_pre_tx_warning", "result": _safe_jsonable(result)}


@app.post("/v1/telco-notify")
def telco_notify(req: TelcoNotifyRequest) -> Dict[str, Any]:
    client_mod = _import_module("mel_dev.features.telco_notification_webhook.src.client")

    incident_obj: Any = req.model_dump()

    # build TelcoIncidentPayload if available
    try:
        schemas_mod = _import_module("mel_dev.features.telco_notification_webhook.src.schemas")
        TelcoIncidentPayload = getattr(schemas_mod, "TelcoIncidentPayload", None)
        if TelcoIncidentPayload is not None:
            incident_obj = TelcoIncidentPayload(**req.model_dump())
    except Exception:
        pass

    client_cls = getattr(client_mod, "TelcoNotificationClient", None)
    if client_cls is not None:
        client = client_cls()
        if hasattr(client, "send_incident"):
            resp = client.send_incident(incident_obj)
            return {"feature": "telco_notification_webhook", "result": _safe_jsonable(resp)}

    resp = _call_fn_or_class_method(client_mod, ["send_incident", "notify_telco", "push_incident"], incident_obj)
    return {"feature": "telco_notification_webhook", "result": _safe_jsonable(resp)}


# -----------------------------------------------------------------------------
# HUAWEI endpoints (ignore Proactive_Warning_Service)
# -----------------------------------------------------------------------------
@app.post("/v1/blacklist/check")
def blacklist_check(req: BlacklistCheckRequest) -> Dict[str, Any]:
    mod = _import_module("HUAWEI.Blacklist_Watchlist_Service.src.blacklist_watchlist_service")
    result = _call_fn_or_class_method(mod, ["check_value", "check", "is_blacklisted", "lookup"], req.value, req.list_type)
    return {"feature": "blacklist_watchlist_service", "result": _safe_jsonable(result)}


@app.post("/v1/phishing-ad-referral/score")
def phishing_ad_referral_score(req: PhishingScoreRequest) -> Dict[str, Any]:
    mod = _import_module("HUAWEI.Phishing_Ad_Referral_Channel_Detector.src.mindspore_detector")
    result = _call_fn_or_class_method(mod, ["predict_url", "predict", "score_url", "detect", "score"], req.url)
    return {"feature": "phishing_ad_referral_channel_detector", "result": _safe_jsonable(result)}
