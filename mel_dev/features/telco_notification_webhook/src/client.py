import time
from typing import Optional, Dict, Any
from datetime import datetime

import requests

from .config import (
    TELCO_WEBHOOK_URL,
    TELCO_WEBHOOK_API_KEY,
    VERIFY_SSL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    SYSTEM_NAME,
    ENVIRONMENT,
)
from .schemas import TelcoIncidentPayload, TelcoWebhookResponse
from .storage import log_incident, log_error


class TelcoNotificationClient:
    """
    Simple authenticated HTTP client for pushing fraud incidents
    to a telco / SIEM / triage webhook.
    """

    def __init__(
        self,
        webhook_url: str = TELCO_WEBHOOK_URL,
        api_key: str = TELCO_WEBHOOK_API_KEY,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        verify_ssl: bool = VERIFY_SSL,
    ):
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-System-Name": SYSTEM_NAME,
            "X-Environment": ENVIRONMENT,
        }

    def send_incident(self, incident) -> TelcoWebhookResponse:
        """
        Sends a single incident to the telco webhook.
        Retries on network errors or 5xx responses.
        Logs both incident and final response for auditing.
        """

        # 🔐 Accept schema object OR dict
        if isinstance(incident, dict):
            payload_dict = incident
        else:
            payload_dict = incident.to_dict()

        attempt = 0
        last_error: Optional[str] = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                resp = requests.post(
                    self.webhook_url,
                    json=payload_dict,
                    headers=self._headers(),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                ok = 200 <= resp.status_code < 300
                telco_ref = resp.headers.get("X-Telco-Ref-Id") or None

                received_at = (
                    datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                )

                response_obj = TelcoWebhookResponse(
                    status_code=resp.status_code,
                    success=ok,
                    telco_reference_id=telco_ref,
                    raw_body=resp.text,
                    received_at=received_at,
                )

                # Log regardless of success
                log_incident(payload_dict, response_obj.to_dict())

                if ok:
                    return response_obj

                last_error = f"Non-2xx status: {resp.status_code}"
            except Exception as e:
                last_error = str(e)
                log_error(
                    {
                        "event": "webhook_exception",
                        "error": last_error,
                        "attempt": attempt,
                        "webhook_url": self.webhook_url,
                    }
                )
                time.sleep(1.0)  # basic backoff

        # Final failure
        received_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        response_obj = TelcoWebhookResponse(
            status_code=0,
            success=False,
            telco_reference_id=None,
            raw_body=last_error,
            received_at=received_at,
        )
        log_incident(payload_dict, response_obj.to_dict())
        return response_obj
