export const ENDPOINTS = {
    HEALTH: {
        SYSTEM: '/health',
        API_V1: '/v1/health',
        PRIVACY: '/v1/governance/privacy/health',
        AUDIT: '/v1/governance/audit/health',
    },
    STATS: {
        AUDIT: '/v1/governance/audit/stats',
        PRIVACY: '/v1/governance/privacy/classifications/stats',
        BLACKLIST: '/v1/blacklist/stats',
        REPUTATION: '/v1/customer-reputation/stats',
        REPUTATION_RECENT_REPORTS: '/v1/customer-reputation/reports/recent',
    },
    FEATURES: {
        CALL_DEFENSE: '/v1/call-triggered-defense',
        PHISHING_SCORE: '/v1/phishing-ad-referral/score',
        CLICK_TX_CORRELATION: '/v1/click-tx-correlation',
        REPUTATION_SUBMIT: '/v1/customer-reputation/report/submit',
        BLACKLIST_CHECK: '/v1/blacklist/check',
        AGENT_RISK: (id: string) => `/v1/customer-reputation/agent/${id}/risk`,
        MERCHANT_RISK: (id: string) => `/v1/customer-reputation/merchant/${id}/risk`,
        ALERTS: '/v1/customer-reputation/alerts',
        PRE_TX_WARNING: '/v1/proactive-pre-tx-warning',
        TELCO_NOTIFY: '/v1/telco-notify',
        SMS_ALERT: '/v1/user-sms-alert',
        ORCHESTRATOR: '/v1/orchestrate',
    },
    GOVERNANCE: {
        PRIVACY_CLASSIFY: '/v1/governance/privacy/classify',
        PRIVACY_ANONYMIZE: '/v1/governance/privacy/anonymize',
        PRIVACY_PII_TYPES: '/v1/governance/privacy/pii-types',
        AUDIT_SEND: '/v1/governance/audit/event',
    },
    BLACKLIST: {
        ADD: '/v1/blacklist/add',
        REMOVE: '/v1/blacklist/remove',
        REBUILD: '/v1/blacklist/rebuild',
    }
};
