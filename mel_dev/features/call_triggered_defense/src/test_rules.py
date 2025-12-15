from .rules import CallTxEvent, high_precision_rule

event = CallTxEvent(
    call_to_tx_delta_seconds=120,
    recipient_first_time=1,
    tx_amount=1500,
)

print("Rule triggered:", high_precision_rule(event))
# Expected output: Rule triggered: True