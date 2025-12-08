from dataclasses import dataclass


@dataclass
class UserRiskProfile:
    """
    Aggregated user profile used by the rule engine for proactive warnings.
    """
    recent_risky_clicks_7d: int
    recent_scam_calls_7d: int
    scam_campaign_intensity: float
    device_age_days: float
    is_new_device: int
    tx_count_total: int
    avg_tx_amount: float
    max_tx_amount: float
    historical_fraud_flag: int
    is_in_campaign_cohort: int
    user_risk_score: float


# Simple rule thresholds (can be tuned)
MIN_CAMPAIGN_INTENSITY = 0.4
MIN_USER_RISK_SCORE = 0.6
MIN_RECENT_ACTIVITY = 1   # at least 1 risky click or scam call


def must_warn_user(profile: UserRiskProfile) -> bool:
    """
    High-precision rule deciding if user MUST receive a proactive warning.

    Fires when:
      - campaign intensity is non-trivial
      - user is in the targeted campaign cohort
      - user risk score is high
      - AND (recent risky activity OR new device OR historical fraud)
    """
    active_campaign = profile.scam_campaign_intensity >= MIN_CAMPAIGN_INTENSITY
    in_cohort = profile.is_in_campaign_cohort == 1
    high_risk_user = profile.user_risk_score >= MIN_USER_RISK_SCORE

    recent_activity = (
        profile.recent_risky_clicks_7d >= MIN_RECENT_ACTIVITY
        or profile.recent_scam_calls_7d >= MIN_RECENT_ACTIVITY
    )
    new_device = profile.is_new_device == 1
    prior_fraud = profile.historical_fraud_flag == 1

    secondary_risk = recent_activity or new_device or prior_fraud

    return bool(active_campaign and in_cohort and high_risk_user and secondary_risk)


def build_warning_message(profile: UserRiskProfile) -> str:
    """
    Generate a user-facing SMS warning template.
    In production, this would be localized and customized.
    """
    base = (
        "We detected an ongoing scam campaign targeting customers like you. "
        "Do NOT share your PIN, OTP or password with anyone. "
        "If you receive a call asking for your balance or PIN, ignore it and call 123."
    )

    if profile.is_new_device:
        extra = " We also noticed a login from a new device; be extra careful."
    elif profile.historical_fraud_flag:
        extra = " Our records show previous suspicious activity; stay vigilant."
    else:
        extra = ""

    return base + extra
