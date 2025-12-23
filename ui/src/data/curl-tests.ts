
export const CURL_TESTS: Record<string, string[]> = {
    call_defense: [
        `curl -X POST "http://127.0.0.1:8000/v1/call-triggered-defense" \\
-H "Content-Type: application/json" \\
-d '{
  "call_id": "call_123456789",
  "caller_number": "+1234567890",
  "receiver_number": "+0987654321",
  "duration_seconds": 45,
  "audio_features": {
    "pitch": 120.5,
    "jitter": 0.02,
    "shimmer": 0.05
  }
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/call-triggered-defense?debug=true" \\
-H "Content-Type: application/json" \\
-d '{
  "call_id": "call_debug_001",
  "caller_number": "+15550000000",
  "receiver_number": "+0987654321",
  "duration_seconds": 120,
  "audio_features": { "pitch": 110.0, "jitter": 0.01, "shimmer": 0.01 }
}'`
    ],
    phishing: [
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://google.com"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://www.ecobank.com"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"http://185.199.108.153/login"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://free-prize-winner.xyz"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://secure-login-account-update.com"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://login.secure.account.verify.update.user.session.info.com/auth"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://secure-bonus-claim-now.net"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://bit.ly/3xYzAbC"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://momo-secure-gh.com/verify"}'`,
        `curl -s -X POST http://127.0.0.1:8000/v1/phishing-ad-referral/score \\
-H "Content-Type: application/json" \\
-d '{"url":"https://wikipedia.org"}'`
    ],
    click_tx: [
        `curl -X POST "http://127.0.0.1:8000/v1/customer-reputation/transactions/click-correlation" \\
-H "Content-Type: application/json" \\
-d '{
  "user_id": "u_998877",
  "session_id": "sess_555",
  "clicks": [ { "element": "login_btn", "timestamp": 1678886400 }, { "element": "transfer_menu", "timestamp": 1678886405 } ]
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \\
  -H "Content-Type: application/json" \\
  -d '{
    "amount": 200,
    "url_risk_score": 0.85,
    "url_reported_flag": 1,
    "clicked_recently": 1,
    "time_between_click_and_tx": 30,
    "time_since_last_tx_seconds": 120,
    "tx_hour": 14,
    "tx_dayofweek": 3,
    "url_hash_numeric": 99999,
    "device_click_count_1d": 5,
    "user_click_count_1d": 2
  }'`,
        `curl -X POST "http://127.0.0.1:8000/v1/click-tx-correlation" \\
  -H "Content-Type: application/json" \\
  -d '{
    "amount": 50,
    "url_risk_score": 0.1,
    "url_reported_flag": 0,
    "clicked_recently": 1,
    "time_between_click_and_tx": 600,
    "time_since_last_tx_seconds": 3600,
    "tx_hour": 10,
    "tx_dayofweek": 1,
    "url_hash_numeric": 1001,
    "device_click_count_1d": 2,
    "user_click_count_1d": 2
  }'`
    ],
    reputation_report: [
        `curl -X POST "http://127.0.0.1:8000/v1/customer-reputation/report/submit" \\
  -H "Content-Type: application/json" \\
  -d '{
    "reporter_id": "user_001",
    "merchant_id": "merchant_abc",
    "report_type": "fraud",
    "rating": 1,
    "title": "Unauthorized charge",
    "description": "I saw a charge I did not authorize.",
    "transaction_id": "txn_1001",
    "amount": 150.00,
    "metadata": {"platform":"mobile","location":"Accra"}
  }'`,
        `curl -X POST "http://127.0.0.1:8000/v1/customer-reputation/report/submit" \\
  -H "Content-Type: application/json" \\
  -d '{
    "reporter_id": "user_002",
    "merchant_id": "merchant_abc",
    "report_type": "scam",
    "rating": 2,
    "title": "Suspicious payment",
    "description": "Merchant requested payment outside platform.",
    "transaction_id": "txn_1002",
    "amount": 300.00,
    "metadata": {"platform":"web","location":"Kumasi"}
  }'`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/report/3505f8ff-9adc-49a7-aa63-ce65578627b1"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/stats"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/merchant/merchant_abc/reports"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/reporter/user_001/reports"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/agent/agent_001/risk"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/merchant/merchant_abc/risk"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/transactions/suspicious?hours=24"`,
        `curl -X GET "http://127.0.0.1:8000/v1/customer-reputation/alerts?threshold=0.7"`
    ],
    blacklist_add: [
        `curl -X POST "http://127.0.0.1:8000/v1/blacklist/add" \\
-H "Content-Type: application/json" \\
-d '{
  "value": "+1234567890",
  "list_type": "phone_number",
  "reason": "Confirmed scammer"
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/blacklist/check" \\
-H "Content-Type: application/json" \\
-d '{
    "entity_id": "+1234567890",
    "entity_type": "phone_number"
}'`,
        `curl -X DELETE "http://127.0.0.1:8000/v1/blacklist/remove" \\
-H "Content-Type: application/json" \\
-d '{
    "value": "+1234567890",
    "list_type": "phone_number"
}'`
    ],
    pre_tx: [
        `curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \\
-H "Content-Type: application/json" \\
-d '{
  "recent_risky_clicks_7d": 5,
  "recent_scam_calls_7d": 2,
  "scam_campaign_intensity": 0.8,
  "device_age_days": 5,
  "is_new_device": 1,
  "tx_count_total": 5,
  "avg_tx_amount": 250,
  "max_tx_amount": 1000,
  "historical_fraud_flag": 1,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.95
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \\
-H "Content-Type: application/json" \\
-d '{
  "recent_risky_clicks_7d": 1,
  "recent_scam_calls_7d": 0,
  "scam_campaign_intensity": 0.2,
  "device_age_days": 300,
  "is_new_device": 0,
  "tx_count_total": 150,
  "avg_tx_amount": 50,
  "max_tx_amount": 200,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 0,
  "user_risk_score": 0.15
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/proactive-pre-tx-warning?debug=true" \\
-H "Content-Type: application/json" \\
-d '{
  "recent_risky_clicks_7d": 0,
  "recent_scam_calls_7d": 2,
  "scam_campaign_intensity": 0.7,
  "device_age_days": 120,
  "is_new_device": 0,
  "tx_count_total": 30,
  "avg_tx_amount": 60,
  "max_tx_amount": 300,
  "historical_fraud_flag": 0,
  "is_in_campaign_cohort": 1,
  "user_risk_score": 0.75
}'`
    ],
    telco_notify: [
        `curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \\
  -H "Content-Type: application/json" \\
  -d '{
    "incident_id": "inc-0001",
    "suspected_number": "0559426442",
    "affected_accounts": ["momo:233559426442"],
    "observed_evidence": {"signal":"multiple scam calls"},
    "timestamps": {"detected_at":"2025-12-15T18:10:00Z"},
    "recommended_action": "INVESTIGATE_AND_TEMP_BLOCK"
  }'`,
        `curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \\
  -H "Content-Type: application/json" \\
  -d '{
    "incident_id": "inc-0002",
    "suspected_number": "0244000000",
    "affected_accounts": ["momo:233244000000","momo:233201111111"],
    "observed_evidence": {
      "cdr_ids": ["cdr_hash_1","cdr_hash_2"],
      "click_hashes": ["urlhash_9"],
      "model_scores": {"proactive_pre_tx_warning": 0.98, "call_triggered_defense": 0.91},
      "rule_flags": {"proactive_pre_tx_warning": true}
    },
    "timestamps": {"window_start":"2025-12-15T10:00:00Z","window_end":"2025-12-15T18:00:00Z"},
    "recommended_action": "TEMP_BLOCK_NUMBER"
  }'`,
        `curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \\
  -H "Content-Type: application/json" \\
  -d '{
    "incident_id": "inc-0003",
    "suspected_number": "0200000000",
    "affected_accounts": [],
    "observed_evidence": {},
    "timestamps": {},
    "recommended_action": "LOG_ONLY"
  }'`,
        `curl -X POST "http://127.0.0.1:8000/v1/telco-notify" \\
  -H "Content-Type: application/json" \\
  -d '{
    "incident_id": "inc-0004",
    "suspected_number": "0550000000",
    "affected_accounts": ["a1","a2","a3","a4","a5"],
    "observed_evidence": {"note":"burst activity"},
    "timestamps": {"detected_at":"2025-12-15T18:12:00Z"},
    "recommended_action": "INVESTIGATE_CAMPAIGN"
  }'`
    ],
    sms_alert: [
        `curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \\
-H "Content-Type: application/json" \\
-d '{
  "ref":"orch-low-001",
  "incident_id":"inc-low-001",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":0,
    "recent_scam_calls_7d":0,
    "scam_campaign_intensity":0.1,
    "device_age_days":365,
    "is_new_device":0,
    "tx_count_total":120,
    "avg_tx_amount":35,
    "max_tx_amount":150,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":0,
    "user_risk_score":0.2
  }
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/orchestrate" \\
-H "Content-Type: application/json" \\
-d '{
  "ref":"orch-high-003",
  "incident_id":"inc-high-003",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "affected_accounts":["momo:233559426442"],
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":2,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":5,
    "is_new_device":1,
    "tx_count_total":3,
    "avg_tx_amount":150,
    "max_tx_amount":600,
    "historical_fraud_flag":1,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.85
  }
}'`,
        `curl -X POST "http://127.0.0.1:8000/v1/orchestrate?debug=true" \\
-H "Content-Type: application/json" \\
-d '{
  "ref":"orch-all-006",
  "incident_id":"inc-all-006",
  "user_phone":"0559426442",
  "suspected_number":"0244000000",
  "affected_accounts":["momo:233559426442"],
  "call_triggered_defense":{
    "tx_amount":3000,
    "recipient_first_time":1,
    "call_to_tx_delta_seconds":30,
    "contact_list_flag":0,
    "nlp_suspicion_score":0.8
  },
  "click_tx_correlation":{
    "tx_amount":1200,
    "url_reported_flag":1,
    "time_between_click_and_tx":20,
    "url_risk_score":0.9,
    "clicked_recently":1,
    "device_click_count_1d":5,
    "user_click_count_1d":3
  },
  "proactive_pre_tx_warning":{
    "recent_risky_clicks_7d":4,
    "recent_scam_calls_7d":2,
    "scam_campaign_intensity":0.9,
    "device_age_days":3,
    "is_new_device":1,
    "tx_count_total":4,
    "avg_tx_amount":120,
    "max_tx_amount":900,
    "historical_fraud_flag":0,
    "is_in_campaign_cohort":1,
    "user_risk_score":0.9
  }
}'`
    ],
    privacy_classify: [
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Email me at test@example.com"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Call +233554123456 now"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Call +233554123456 or email test@example.com"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"User logged in from 197.0.0.1"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/classify" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Payment card 4111 1111 1111 1111 was used"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/anonymize" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Call +233554123456 or email test@example.com","strategy":"mask"}'`,
        `curl -s -X POST "http://127.0.0.1:8000/v1/governance/privacy/anonymize" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"test@example.com +233554123456 197.0.0.1","strategy":"hash"}'`
    ],
    audit_event: [
        `curl -X POST "http://127.0.0.1:8000/v1/governance/audit/event" \\
-H "Content-Type: application/json" \\
-d '{
  "event_type": "data_access",
  "actor_id": "user_123",
  "resource_id": "file_abc",
  "status": "success",
  "details": {"action": "read"}
}'`
    ]
};
