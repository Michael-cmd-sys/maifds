"""
Database schema definitions
"""

CREATE_REPORTS_TABLE = """
CREATE TABLE IF NOT EXISTS reports (
    report_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    reporter_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    report_type TEXT NOT NULL,
    rating INTEGER,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    transaction_id TEXT,
    amount REAL,
    metadata_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_REPORTERS_TABLE = """
CREATE TABLE IF NOT EXISTS reporters (
    reporter_id TEXT PRIMARY KEY,
    credibility_score REAL DEFAULT 0.5,
    total_reports INTEGER DEFAULT 0,
    verified_reports INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MERCHANTS_TABLE = """
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id TEXT PRIMARY KEY,
    merchant_name TEXT,
    total_reports INTEGER DEFAULT 0,
    average_rating REAL,
    reputation_score REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_AGENTS_TABLE = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    agent_name TEXT,
    credibility_score REAL DEFAULT 0.5,
    risk_score REAL DEFAULT 0.5,
    total_recruits INTEGER DEFAULT 0,
    active_merchants INTEGER DEFAULT 0,
    network_depth INTEGER DEFAULT 0,
    recruitment_rate REAL DEFAULT 0.0,
    avg_transaction_amount REAL DEFAULT 0.0,
    suspicious_activity_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_AGENT_NETWORKS_TABLE = """
CREATE TABLE IF NOT EXISTS agent_networks (
    network_id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    strength_score REAL DEFAULT 0.0,
    transaction_count INTEGER DEFAULT 0,
    total_amount REAL DEFAULT 0.0,
    risk_level TEXT DEFAULT 'medium',
    first_interaction DATETIME,
    last_interaction DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
    FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
);
"""

CREATE_MULE_ACCOUNTS_TABLE = """
CREATE TABLE IF NOT EXISTS mule_accounts (
    account_id TEXT PRIMARY KEY,
    account_type TEXT NOT NULL,
    mule_score REAL DEFAULT 0.0,
    network_id TEXT,
    transaction_patterns TEXT,
    risk_indicators TEXT,
    is_confirmed_mule BOOLEAN DEFAULT FALSE,
    detection_date DATETIME,
    rapid_transaction_count INTEGER DEFAULT 0,
    circular_transaction_count INTEGER DEFAULT 0,
    avg_hold_time_minutes REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (network_id) REFERENCES agent_networks(network_id)
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_merchant ON reports(merchant_id);",
    "CREATE INDEX IF NOT EXISTS idx_reporter ON reports(reporter_id);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON reports(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_report_type ON reports(report_type);",
    "CREATE INDEX IF NOT EXISTS idx_agent_network_agent ON agent_networks(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_network_merchant ON agent_networks(merchant_id);",
    "CREATE INDEX IF NOT EXISTS idx_agent_network_network ON agent_networks(network_id);",
    "CREATE INDEX IF NOT EXISTS idx_mule_network ON mule_accounts(network_id);",
    "CREATE INDEX IF NOT EXISTS idx_mule_score ON mule_accounts(mule_score);",
    # "CREATE INDEX IF NOT EXISTS idx_agent_risk_score ON agents(risk_score);",
]

CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    entity_id TEXT,
    entity_name TEXT,
    risk_score REAL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME
);
"""

ALL_SCHEMAS = [
    CREATE_REPORTS_TABLE,
    CREATE_REPORTERS_TABLE,
    CREATE_MERCHANTS_TABLE,
    CREATE_AGENTS_TABLE,
    CREATE_AGENT_NETWORKS_TABLE,
    CREATE_MULE_ACCOUNTS_TABLE,
    CREATE_ALERTS_TABLE,
] + CREATE_INDEXES
