from schemas import IncidentEvidence, TelcoIncidentPayload
from client import TelcoNotificationClient
from config import SYSTEM_NAME, ENVIRONMENT


def main():
    # Example: triggered by a high-confidence scam caller detection
    evidence = IncidentEvidence(
        cdr_ids=["cdr_hash_abc123", "cdr_hash_def456"],
        click_hashes=["url_hash_111", "url_hash_222"],
        model_scores={
            "call_triggered_defense": 0.97,
            "click_tx_link_correlation": 0.91,
        },
        rule_flags={
            "call_triggered_defense_rule": True,
            "click_tx_rule": True,
        },
    )

    payload = TelcoIncidentPayload.new(
        suspected_number="+233201234567",           # simulated scam caller
        affected_accounts=["ACC-001234", "ACC-009876"],
        recommended_action="TEMP_BLOCK_NUMBER",     # could also be "INVESTIGATE_CAMPAIGN"
        severity="HIGH",
        source_system=SYSTEM_NAME,
        environment=ENVIRONMENT,
        evidence=evidence,
        metadata={
            "note": "Simulated incident from test_client.py",
            "suggested_next_steps": [
                "Trace call origin",
                "Monitor number across network",
                "Review customer complaints for this MSISDN",
            ],
        },
    )

    client = TelcoNotificationClient()
    response = client.send_incident(payload)

    print("Webhook response:")
    print(response.to_dict())


if __name__ == "__main__":
    main()
