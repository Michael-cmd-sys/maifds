"""
Database schema definitions for audit trail and compliance
"""

# Audit Events Table
CREATE_AUDIT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    event_type TEXT NOT NULL,
    component TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    entity_id TEXT,
    entity_type TEXT,
    decision_data TEXT,
    explanation_data TEXT,
    privacy_impact TEXT NOT NULL DEFAULT 'none',
    ip_address TEXT,
    user_agent TEXT,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# User Consent Table
CREATE_USER_CONSENT_TABLE = """
CREATE TABLE IF NOT EXISTS user_consent (
    consent_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    consent_type TEXT NOT NULL,
    granted BOOLEAN NOT NULL,
    timestamp DATETIME NOT NULL,
    ip_address TEXT,
    user_agent TEXT,
    expires_at DATETIME,
    purpose TEXT,
    legal_basis TEXT,
    withdrawn_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Data Retention Table
CREATE_DATA_RETENTION_TABLE = """
CREATE TABLE IF NOT EXISTS data_retention (
    retention_id TEXT PRIMARY KEY,
    data_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_table TEXT NOT NULL,
    retention_days INTEGER NOT NULL,
    delete_after DATETIME NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    reason TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Privacy Requests Table
CREATE_PRIVACY_REQUESTS_TABLE = """
CREATE TABLE IF NOT EXISTS privacy_requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    request_type TEXT NOT NULL,
    request_data TEXT,
    processing_status TEXT NOT NULL DEFAULT 'pending',
    completion_timestamp DATETIME,
    data_deleted_count INTEGER DEFAULT 0,
    data_exported_count INTEGER DEFAULT 0,
    export_path TEXT,
    ip_address TEXT,
    user_agent TEXT,
    assigned_to TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Model Explanations Table
CREATE_MODEL_EXPLANATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS model_explanations (
    explanation_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT,
    prediction_id TEXT NOT NULL,
    input_features TEXT NOT NULL,
    prediction TEXT NOT NULL,
    prediction_probability REAL,
    feature_importance TEXT,
    explanation_method TEXT,
    explanation_summary TEXT,
    counterfactual_analysis TEXT,
    processing_time_ms INTEGER,
    component TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Decision Appeals Table
CREATE_DECISION_APPEALS_TABLE = """
CREATE TABLE IF NOT EXISTS decision_appeals (
    appeal_id TEXT PRIMARY KEY,
    original_decision_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    appeal_reason TEXT NOT NULL,
    appeal_status TEXT NOT NULL DEFAULT 'pending',
    review_notes TEXT,
    reviewer_id TEXT,
    review_timestamp DATETIME,
    outcome TEXT,
    new_decision_data TEXT,
    ip_address TEXT,
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Access Control Table
CREATE_ACCESS_CONTROL_TABLE = """
CREATE TABLE IF NOT EXISTS access_control (
    access_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT,
    permission_level TEXT NOT NULL,
    granted_by TEXT NOT NULL,
    granted_at DATETIME NOT NULL,
    expires_at DATETIME,
    conditions TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used DATETIME,
    usage_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Data Classification Table
CREATE_DATA_CLASSIFICATION_TABLE = """
CREATE TABLE IF NOT EXISTS data_classification (
    classification_id TEXT PRIMARY KEY,
    data_type TEXT NOT NULL,
    sensitivity_level TEXT NOT NULL,
    retention_period_days INTEGER NOT NULL,
    access_requirements TEXT,
    encryption_required BOOLEAN NOT NULL DEFAULT FALSE,
    audit_level TEXT NOT NULL DEFAULT 'standard',
    legal_restrictions TEXT,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# Audit Indexes
CREATE_AUDIT_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type);",
    "CREATE INDEX IF NOT EXISTS idx_audit_component ON audit_events(component);",
    "CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_audit_entity_id ON audit_events(entity_id);",
    "CREATE INDEX IF NOT EXISTS idx_audit_privacy_impact ON audit_events(privacy_impact);",
    "CREATE INDEX IF NOT EXISTS idx_consent_user_id ON user_consent(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_consent_type ON user_consent(consent_type);",
    "CREATE INDEX IF NOT EXISTS idx_retention_delete_after ON data_retention(delete_after);",
    "CREATE INDEX IF NOT EXISTS idx_retention_status ON data_retention(status);",
    "CREATE INDEX IF NOT EXISTS idx_privacy_request_user_id ON privacy_requests(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_privacy_request_status ON privacy_requests(processing_status);",
    "CREATE INDEX IF NOT EXISTS idx_explanation_prediction_id ON model_explanations(prediction_id);",
    "CREATE INDEX IF NOT EXISTS idx_explanation_model_name ON model_explanations(model_name);",
    "CREATE INDEX IF NOT EXISTS idx_appeal_user_id ON decision_appeals(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_appeal_status ON decision_appeals(appeal_status);",
    "CREATE INDEX IF NOT EXISTS idx_access_user_id ON access_control(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_access_resource ON access_control(resource_type, resource_id);",
    "CREATE INDEX IF NOT EXISTS idx_classification_data_type ON data_classification(data_type);"
]

# All schemas combined
ALL_AUDIT_SCHEMAS = [
    CREATE_AUDIT_EVENTS_TABLE,
    CREATE_USER_CONSENT_TABLE,
    CREATE_DATA_RETENTION_TABLE,
    CREATE_PRIVACY_REQUESTS_TABLE,
    CREATE_MODEL_EXPLANATIONS_TABLE,
    CREATE_DECISION_APPEALS_TABLE,
    CREATE_ACCESS_CONTROL_TABLE,
    CREATE_DATA_CLASSIFICATION_TABLE
] + CREATE_AUDIT_INDEXES