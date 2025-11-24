from inference import run_inference

fake_event = {
    "amount": 250000,
    "transaction_day": 2,
    "transaction_hour": 15,
    "origin_balance_delta": -250000,
    "dest_balance_delta": 250000,
    "is_large_amount": 1,
    "recipient_first_time": 1,
    "oldbalanceOrg": 500000,
    "newbalanceOrig": 250000,
    "oldbalanceDest": 100000,
    "newbalanceDest": 350000,

    "has_recent_call": 1,
    "call_to_tx_delta_seconds": 45,
    "call_duration_seconds": 60,
    "contact_list_flag": 0,
    "device_age_days": 120,
    "nlp_suspicion_score": 0.92,
}

result = run_inference(fake_event)
print(result)

# Expected JSON output:
#{
#  "fraud_probability": 0.87,
#  "rule_flag": true,
#  "risk_level": "HIGH",
#  "reason": "Rule-based high-risk pattern detected",
#  "actions": [
#    "SMS_USER_ALERT",
#    "TEMP_HOLD",
#    "NOTIFY_TELCO"
#  ]
#}
