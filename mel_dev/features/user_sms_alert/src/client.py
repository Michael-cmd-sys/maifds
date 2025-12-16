import time
from typing import Optional, Dict, Any
import requests

from .config import (
    MOOLRE_SMS_SEND_URL,
    MOOLRE_X_API_VASKEY,
    MOOLRE_SENDER_ID,
    VERIFY_SSL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
)
from .schemas import SmsSendResult
from .storage import log_sms_sent, log_sms_error

class MoolreSmsClient:
    def __init__(
        self,
        sender_id: str = MOOLRE_SENDER_ID,
        api_key: str = MOOLRE_X_API_VASKEY,
        url: str = MOOLRE_SMS_SEND_URL,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        verify_ssl: bool = VERIFY_SSL,
    ):
        self.sender_id = sender_id
        self.api_key = api_key
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "X-API-VASKEY": self.api_key,
        }
        
        return h

    def send_sms(self, recipient: str, message: str, ref: Optional[str] = None) -> SmsSendResult:
        if not self.api_key:
            raise ValueError("MOOLRE_X_API_VASKEY is missing in environment/.env")
        if not self.sender_id:
            raise ValueError("MOOLRE_SENDER_ID is missing in environment/.env")

        payload = {
            "type": 1,
            "senderid": self.sender_id,
            "messages": [{
                "recipient": recipient,
                "message": message,
                "ref": ref or ""
            }]
        }

        attempt = 0
        last_error: Optional[str] = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                resp = requests.post(
                    self.url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                ok = 200 <= resp.status_code < 300
                raw = resp.text

                provider_code = None
                provider_msg = None
                try:
                    data = resp.json()
                    provider_code = data.get("code")
                    provider_msg = data.get("message")
                except Exception:
                    pass

                result = SmsSendResult(
                    status_code=resp.status_code,
                    success=ok,
                    provider_code=provider_code,
                    provider_message=provider_msg,
                    raw_body=raw,
                    request_id=ref,
                )

                # Always log for dashboard visibility
                log_sms_sent(payload, result.to_dict())

                if ok:
                    return result

                last_error = f"Non-2xx status: {resp.status_code}"
            except Exception as e:
                last_error = str(e)
                log_sms_error({
                    "event": "moolre_exception",
                    "error": last_error,
                    "attempt": attempt,
                    "url": self.url,
                })
                time.sleep(1.0)

        # Final failure
        result = SmsSendResult(
            status_code=0,
            success=False,
            provider_code=None,
            provider_message=None,
            raw_body=last_error,
            request_id=ref,
        )
        log_sms_sent(payload, result.to_dict())
        return result
