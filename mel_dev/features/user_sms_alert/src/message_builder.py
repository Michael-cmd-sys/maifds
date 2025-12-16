from typing import Dict, Any

def build_user_sms(threat_type: str, context: Dict[str, Any]) -> str:
    """
    threat_type examples: 'OTP_SCAM', 'CALL_TO_TX', 'PHISHING_LINK', 'GENERAL'
    context can include: suspected_number, tips, hotline, etc.
    """
    hotline = context.get("hotline", "123")
    suspected = context.get("suspected_number", "")

    if threat_type == "OTP_SCAM":
        return (
            "MAIFDS ALERT: Possible OTP/PIN scam detected. Do NOT share your OTP/PIN/password. "
            f"If you were contacted by {suspected}, hang up and call {hotline}."
        )

    if threat_type == "PHISHING_LINK":
        return (
            "MAIFDS ALERT: Dangerous link/phishing activity detected. Do NOT click unknown links. "
            f"If you interacted with {suspected}, secure your account and call {hotline}."
        )

    if threat_type == "CALL_TO_TX":
        return (
            "MAIFDS ALERT: Suspicious call-to-transaction pattern detected. "
            "Do NOT send money under pressure. Verify the caller independently."
        )

    # GENERAL
    return (
        "MAIFDS ALERT: Suspicious activity detected on your number. "
        "Do NOT share PIN/OTP. Verify requests before sending money."
    )
