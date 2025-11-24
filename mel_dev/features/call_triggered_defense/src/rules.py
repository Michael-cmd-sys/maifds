from dataclasses import dataclass

# Thresholds — these can be tuned later
CALL_TO_TX_MAX_SECONDS = 300        # 5 minutes
TX_AMOUNT_THRESHOLD = 1000.0        # amount threshold for high-risk
NLP_SUSPICION_THRESHOLD = 0.8       # optional, if NLP score is used


@dataclass
class CallTxEvent:
    """
    Represents a single transaction + the most recent call before it.
    """
    call_to_tx_delta_seconds: float | None
    recipient_first_time: int
    tx_amount: float
    contact_list_flag: int = 0
    nlp_suspicion_score: float | None = None


def high_precision_rule(event: CallTxEvent) -> bool:
    """
    Rule-based high precision logic:
    Flags only when the pattern is very suspicious.
    This minimizes false positives.
    """

    suspicious_delta = (
        event.call_to_tx_delta_seconds is not None 
        and event.call_to_tx_delta_seconds < CALL_TO_TX_MAX_SECONDS
    )

    big_amount = event.tx_amount > TX_AMOUNT_THRESHOLD
    new_recipient = event.recipient_first_time == 1

    # Optional NLP-based suspicion
    nlp_flag = (
        event.nlp_suspicion_score is not None 
        and event.nlp_suspicion_score > NLP_SUSPICION_THRESHOLD
    )

    # Core rule:
    if suspicious_delta and new_recipient and big_amount:
        return True

    # Additional rule using NLP score
    if suspicious_delta and nlp_flag:
        return True

    return False
