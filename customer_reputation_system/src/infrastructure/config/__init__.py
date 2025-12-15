"""Configuration module for the customer reputation system."""

try:
    from .base import Settings, settings
except ImportError:
    Settings = None
    settings = None

try:
    from .database import DatabaseSettings, database_settings
except ImportError:
    DatabaseSettings = None
    database_settings = None

try:
    from .nlp import NLPSettings, nlp_settings
except ImportError:
    NLPSettings = None
    nlp_settings = None

try:
    from .credibility import CredibilitySettings, credibility_settings
except ImportError:
    CredibilitySettings = None
    credibility_settings = None

try:
    from .reputation import ReputationSettings, reputation_settings
except ImportError:
    ReputationSettings = None
    reputation_settings = None

__all__ = [
    "Settings",
    "DatabaseSettings", 
    "NLPSettings",
    "CredibilitySettings",
    "ReputationSettings",
    "settings",
    "database_settings",
    "nlp_settings",
    "credibility_settings",
    "reputation_settings"
]