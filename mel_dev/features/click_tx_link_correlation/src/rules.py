from dataclasses import dataclass


@dataclass
class ClickTxEvent:
    """
    Event object for rule-based evaluation of click→transaction risk.
    """
    time_between_click_and_tx: float        # seconds
    url_risk_score: float                   # 0–1
    url_reported_flag: int                  # 0/1
    amount: float                           # transaction amount
    median_amount: float                    # reference median for 'large' tx
    clicked_recently: int                   # 0/1
    device_click_count_1d: int = 0
    user_click_count_1d: int = 0


# Rule thresholds (tuneable)
RECENT_CLICK_THRESHOLD_SECONDS = 300.0      # 5 minutes
HIGH_URL_RISK_THRESHOLD = 0.7
LARGE_AMOUNT_MULTIPLIER = 2.0               # large = >= 2x median


def high_precision_click_rule(event: ClickTxEvent) -> bool:
    """
    High-precision rule:
    Fire ONLY when a transaction is clearly tied to a dangerous link.

    Conditions (AND):
      - click was very recent
      - URL risk score is high OR URL is in hard blacklist
      - transaction is large vs user's typical behaviour
    """
    recent = (
        event.clicked_recently == 1
        or event.time_between_click_and_tx <= RECENT_CLICK_THRESHOLD_SECONDS
    )

    high_risk_url = (
        event.url_risk_score >= HIGH_URL_RISK_THRESHOLD
        or event.url_reported_flag == 1
    )

    large_amount = event.amount >= (LARGE_AMOUNT_MULTIPLIER * event.median_amount)

    return bool(recent and high_risk_url and large_amount)
