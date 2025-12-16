from __future__ import annotations
from typing import Any, Dict
from uuid import uuid4

from mel_dev.features.call_triggered_defense.src.inference import run_inference as call_defense
from mel_dev.features.click_tx_link_correlation.src.inference import run_inference as click_tx
from mel_dev.features.proactive_pre_tx_warning.src.inference import run_inference as proactive_warn

from mel_dev.features.telco_notification_webhook.src.client import TelcoNotificationClient
from mel_dev.features.user_sms_alert.src.client import MoolreSmsClient
from mel_dev.features.user_sms_alert.src.message_builder import build_user_sms
from mel_dev.features.user_sms_alert.src.phone import normalize_gh_number

# Import logging functions
from mel_dev.features.orchestrator.src.storage import log_orchestrator_run, log_orchestrator_error


def decide_actions(results: Dict[str, Any]) -> Dict[str, Any]:
    risk_levels = []
    for _, v in results.items():
        if isinstance(v, dict):
            risk_levels.append(v.get("risk_level") or v.get("risk_tier") or "LOW")

    risk_levels = [str(r).upper() for r in risk_levels]
    overall = "LOW"
    if "CRITICAL" in risk_levels:
        overall = "CRITICAL"
    elif "HIGH" in risk_levels:
        overall = "HIGH"
    elif "MEDIUM" in risk_levels:
        overall = "MEDIUM"

    return {
        "overall_risk": overall,
        "send_user_sms": overall in ("MEDIUM", "HIGH", "CRITICAL"),
        "send_telco_webhook": overall in ("HIGH", "CRITICAL"),
    }


def run_orchestrator(event: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
    try:
        results: Dict[str, Any] = {}

        payload = event.get("call_triggered_defense")
        if isinstance(payload, dict) and payload:
            results["call_triggered_defense"] = call_defense(payload, debug=debug)

        payload = event.get("click_tx_correlation")
        if isinstance(payload, dict) and payload:
            results["click_tx_correlation"] = click_tx(payload, debug=debug)

        payload = event.get("proactive_pre_tx_warning")
        if isinstance(payload, dict) and payload:
            results["proactive_pre_tx_warning"] = proactive_warn(payload, debug=debug)

        decision = decide_actions(results)
        actions_done = {"sms": None, "telco": None}

        # 1) USER SMS
        if decision["send_user_sms"] and event.get("user_phone"):
            phone_norm = normalize_gh_number(event["user_phone"])

            sms_text = build_user_sms(
                decision["overall_risk"],
                {
                    "suspected_number": event.get("suspected_number"),
                    "features": results,
                },
            )

            # ✅ ALWAYS UNIQUE ref (prevents MOOLRE ASMS05)
            base_ref = event.get("ref") or event.get("incident_id") or "maifds"
            sms_ref = f"{base_ref}-{uuid4().hex[:6]}"

            sms_client = MoolreSmsClient()
            sms_resp = sms_client.send_sms(
                recipient=phone_norm,
                message=sms_text,
                ref=sms_ref,
            )

            actions_done["sms"] = {
                "recipient": phone_norm,
                "sms_text": sms_text,
                "ref": sms_ref,
                "provider": sms_resp.to_dict(),
            }

        # 2) TELCO WEBHOOK
        if decision["send_telco_webhook"] and event.get("suspected_number"):
            telco_client = TelcoNotificationClient()

            incident = {
                "incident_id": event.get("incident_id") or f"auto-{uuid4().hex[:12]}",
                "suspected_number": event.get("suspected_number"),
                "affected_accounts": event.get("affected_accounts", []),
                "observed_evidence": {"feature_results": results},
                "timestamps": event.get("timestamps", {}),
                "recommended_action": "INVESTIGATE_AND_TEMP_BLOCK",
            }

            telco_resp = telco_client.send_incident(incident)
            actions_done["telco"] = telco_resp.to_dict()

        out = {
            "decision": decision,
            "feature_results": results,
            "actions_done": actions_done,
        }

        # ✅ log for dashboard
        log_orchestrator_run(request_payload=event, result=out)

        return out

    except Exception as e:
        # ✅ error log for dashboard/debugging
        log_orchestrator_error(request_payload=event, error=str(e))
        raise
