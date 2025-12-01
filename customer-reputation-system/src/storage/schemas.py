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

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_merchant ON reports(merchant_id);",
    "CREATE INDEX IF NOT EXISTS idx_reporter ON reports(reporter_id);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON reports(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_report_type ON reports(report_type);",
]

ALL_SCHEMAS = [
    CREATE_REPORTS_TABLE,
    CREATE_REPORTERS_TABLE,
    CREATE_MERCHANTS_TABLE,
] + CREATE_INDEXES
