from inference import run_inference


def main():
    """
    Build a synthetic high-risk event and run full inference.
    Adjust numbers if you want to test different scenarios.
    """
    example_event = {
    "amount": 2000.0,                 # very large tx
    "time_since_last_tx_seconds": 10.0,
    "tx_hour": 23,
    "tx_dayofweek": 5,

    "url_hash_numeric": 1234567890,
    "url_risk_score": 0.99,           # very risky URL
    "url_reported_flag": 1,           # in blacklist
    "clicked_recently": 1,            # recent click
    "time_between_click_and_tx": 30.0,

    "device_click_count_1d": 50,
    "user_click_count_1d": 50,
}


    result = run_inference(example_event)
    print(result)


if __name__ == "__main__":
    main()
