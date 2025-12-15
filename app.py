"""
maifds_repo/app.py

Unified FastAPI gateway for all Fraud Defense features in this repo.

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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


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
            # allow default ctor only
                inst = obj()
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
    tx_amount: float
    recipient_first_time: int = 0
    call_to_tx_delta_seconds: float


class ClickTxCorrelationRequest(BaseModel):
    tx_amount: float
    url_reported_flag: int
    time_between_click_and_tx: float


class ProactiveWarningRequest(BaseModel):
    user_id: Optional[str] = None
    tx_amount: float = 0
    recently_clicked: int = 0
    recently_called: int = 0
    active_campaign_indicator: int = 0
    new_device_flag: int = 0


class TelcoNotifyRequest(BaseModel):
    incident_id: str
    suspected_number: str
    affected_accounts: List[str] = Field(default_factory=list)
    observed_evidence: Dict[str, Any] = Field(default_factory=dict)
    timestamps: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str


class BlacklistCheckRequest(BaseModel):
    value: str
    list_type: str


class PhishingScoreRequest(BaseModel):
    url: str


# -----------------------------------------------------------------------------
# mel_dev endpoints (your 4)
# -----------------------------------------------------------------------------
@app.post("/v1/call-triggered-defense")
def call_triggered_defense(req: CallTriggeredDefenseRequest) -> Dict[str, Any]:
    mod = _import_module("mel_dev.features.call_triggered_defense.src.inference")

    payload = {
        "tx_amount": req.tx_amount,
        "recipient_first_time": req.recipient_first_time,
        "call_to_tx_delta_seconds": req.call_to_tx_delta_seconds,
    }

    result = _call_fn_or_class_method(mod, ["run_inference", "predict", "infer", "score"], payload)
    return {"feature": "call_triggered_defense", "result": _safe_jsonable(result)}


@app.post("/v1/click-tx-correlation")
def click_tx_correlation(req: ClickTxCorrelationRequest) -> Dict[str, Any]:
    mod = _import_module("mel_dev.features.click_tx_link_correlation.src.inference")

    payload = req.model_dump()
    result = _call_fn_or_class_method(mod, ["run_inference", "predict", "infer", "score"], payload)
    return {"feature": "click_tx_link_correlation", "result": _safe_jsonable(result)}


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
