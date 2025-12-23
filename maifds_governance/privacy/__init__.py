"""
Privacy Control Framework

Comprehensive privacy management system for fraud detection with:
- Data anonymization and pseudonymization
- Consent management and tracking
- GDPR compliance features
- Right to be forgotten implementation
- Data classification and access control
"""

from .anonymizer import DataAnonymizer
from .consent_manager import ConsentManager
from .gdpr_compliance import GDPRComplianceManager
from .access_control import PrivacyAccessController
from .data_classifier import DataClassifier
from dataclasses import asdict

__all__ = [
    'DataAnonymizer',
    'ConsentManager', 
    'GDPRComplianceManager',
    'PrivacyAccessController',
    'DataClassifier'
]

# ---------------------------------------------------------------------
# FastAPI gateway wrappers (stable function entrypoints)
# ---------------------------------------------------------------------
from typing import Any, Dict, Optional
import logging

from .data_classifier import DataClassifier
from .anonymizer import DataAnonymizer
from .consent_manager import ConsentManager
from .gdpr_compliance import GDPRComplianceManager
from .access_control import PrivacyAccessController

logger = logging.getLogger(__name__)

_CLASSIFIER: Optional[DataClassifier] = None
_ANONYMIZER: Optional[DataAnonymizer] = None
_CONSENT: Optional[ConsentManager] = None
_GDPR: Optional[GDPRComplianceManager] = None
_ACCESS: Optional[PrivacyAccessController] = None


def _get_services() -> Dict[str, Any]:
    global _CLASSIFIER, _ANONYMIZER, _CONSENT, _GDPR, _ACCESS
    if _CLASSIFIER is None:
        _CLASSIFIER = DataClassifier()
    if _ANONYMIZER is None:
        _ANONYMIZER = DataAnonymizer()
    if _CONSENT is None:
        _CONSENT = ConsentManager()
    if _GDPR is None:
        _GDPR = GDPRComplianceManager()
    if _ACCESS is None:
        _ACCESS = PrivacyAccessController()

    return {
        "classifier": _CLASSIFIER,
        "anonymizer": _ANONYMIZER,
        "consent": _CONSENT,
        "gdpr": _GDPR,
        "access": _ACCESS,
    }


def _call_method_strict(obj: Any, method: str, *args, **kwargs) -> Any:
    fn = getattr(obj, method, None)
    if not callable(fn):
        public = [m for m in dir(obj) if not m.startswith("_")]
        raise AttributeError(
            f"{obj.__class__.__name__} has no method '{method}'. Available: {public}"
        )
    return fn(*args, **kwargs)


def classify_text(text: str, method: str | None = None) -> Dict[str, Any]:
    """
    Gateway wrapper used by FastAPI.
    DataClassifier does NOT have `classify()`. It has `classify_data()`.
    """
    services = _get_services()
    svc = services["classifier"]

    # Default to the real method
    use_method = (method or "classify_data").strip()

    # Backward compatibility if something sends "classify"
    if use_method == "classify":
        use_method = "classify_data"

    # If still invalid, force safe default
    if not hasattr(svc, use_method):
        use_method = "classify_data"

    return _call_method_strict(svc, use_method, text)


def anonymize_text(text: str, strategy: str = "text", method: str | None = None) -> Dict[str, Any]:
    """
    FastAPI wrapper.
    DataAnonymizer real method for a single string is anonymize_field(value, field_type='text').
    We map 'strategy' -> 'field_type' (email, phone, ip_address, credit_card, name, address, text).
    """
    svc = _get_services()["anonymizer"]
    use_method = method or "anonymize_field"
    return _call_method_strict(svc, use_method, text, strategy)


def health() -> dict:
    s = _get_services()
    classifier = s["classifier"]
    anonymizer = s["anonymizer"]

    classifier_methods = [m for m in dir(classifier) if not m.startswith("_") and callable(getattr(classifier, m))]
    anonymizer_methods = [m for m in dir(anonymizer) if not m.startswith("_") and callable(getattr(anonymizer, m))]

    return {
        "status": "ok",
        "classifier_methods": sorted(classifier_methods),
        "anonymizer_methods": sorted(anonymizer_methods),
    }


from dataclasses import asdict

def _to_jsonable_classification(c):
    d = asdict(c)
    d["category"] = c.category.value
    d["sensitivity"] = c.sensitivity.value
    d["regulatory_scope"] = [r.value for r in c.regulatory_scope]
    d["classification_date"] = c.classification_date.isoformat()
    return d


def _classification_to_dict(c):
    d = asdict(c)
    d["category"] = c.category.value
    d["sensitivity"] = c.sensitivity.value
    d["regulatory_scope"] = [r.value for r in c.regulatory_scope]
    d["classification_date"] = c.classification_date.isoformat()
    return d