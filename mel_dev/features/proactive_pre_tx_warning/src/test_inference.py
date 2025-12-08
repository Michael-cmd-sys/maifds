from inference import run_inference


def main():
    """
    Simulate a user during an active scam campaign and see if they should be warned.
    """
    example_user = {
        # recent risky activity
        "recent_risky_clicks_7d": 3,
        "recent_scam_calls_7d": 2,

        # campaign-level signal
        "scam_campaign_intensity": 0.8,
        "is_in_campaign_cohort": 1,

        # device & transaction behaviour
        "device_age_days": 10.0,
        "is_new_device": 1,
        "tx_count_total": 15,
        "avg_tx_amount": 200.0,
        "max_tx_amount": 1000.0,
        "historical_fraud_flag": 1,

        # composite risk score (from pipeline)
        "user_risk_score": 0.85,
    }

    result = run_inference(example_user)
    print(result)


if __name__ == "__main__":
    main()
