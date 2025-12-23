"""
customer_reputation_system/src/ingestion/report_handler.py

Main report ingestion handler.

Responsibilities:
- Validate incoming report payloads (Pydantic)
- Persist raw JSON backup to disk (best-effort)
- Insert report into SQLite via DatabaseManager
- Optionally run NLP text analysis and attach results (best-effort)
- Optionally update reporter credibility and merchant reputation (best-effort)

This module is written to work from the unified repo root (FastAPI gateway),
so imports are package-qualified (customer_reputation_system.src...).
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
from pydantic import ValidationError

# ---------------------------------------------------------------------
# Imports: always package-qualified (do NOT use "from src....")
# ---------------------------------------------------------------------
from customer_reputation_system.src.models.report_model import Report, ReportMetadata
from customer_reputation_system.src.storage.database import DatabaseManager

# Logging + settings are in customer_reputation_system/config in your tree
from customer_reputation_system.config.logging_config import setup_logger

logger = setup_logger(__name__)

# RAW_DATA_DIR is optional – if not present, we fallback gracefully.
try:
    from customer_reputation_system.config.settings import RAW_DATA_DIR  # type: ignore
except Exception:
    RAW_DATA_DIR = None  # type: ignore


# ---------------------------------------------------------------------
# Optional feature modules (NLP / Credibility / Reputation)
# ---------------------------------------------------------------------
try:
    from customer_reputation_system.src.nlp.text_analyzer import TextAnalyzer  # type: ignore

    NLP_AVAILABLE = True
except Exception:
    TextAnalyzer = None  # type: ignore
    NLP_AVAILABLE = False
    logger.warning("NLP module not available. Text analysis will be skipped.")

try:
    from customer_reputation_system.src.credibility.calculator import CredibilityCalculator  # type: ignore

    CREDIBILITY_AVAILABLE = True
except Exception:
    CredibilityCalculator = None  # type: ignore
    CREDIBILITY_AVAILABLE = False
    logger.warning("Credibility module not available. Credibility scoring will be skipped.")

try:
    from customer_reputation_system.src.reputation.calculator import ReputationCalculator  # type: ignore

    REPUTATION_AVAILABLE = True
except Exception:
    ReputationCalculator = None  # type: ignore
    REPUTATION_AVAILABLE = False
    logger.warning("Reputation module not available. Reputation scoring will be skipped.")


# ---------------------------------------------------------------------
# Helper: safe JSON serialization
# ---------------------------------------------------------------------
def _json_default(x: Any) -> str:
    try:
        return str(x)
    except Exception:
        return "<non-serializable>"


def _safe_dict(x: Any) -> Dict[str, Any]:
    """
    Convert known model objects to dict without crashing.
    """
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if hasattr(x, "to_dict") and callable(getattr(x, "to_dict")):
        try:
            return x.to_dict()
        except Exception:
            pass
    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
        try:
            return x.model_dump()
        except Exception:
            pass
    if hasattr(x, "__dict__"):
        try:
            return dict(x.__dict__)
        except Exception:
            pass
    try:
        return asdict(x)  # dataclass
    except Exception:
        return {"value": str(x)}


# ---------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------
class ReportHandler:
    """Handles report submission and ingestion."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None, enable_nlp: bool = True):
        """
        Args:
            db_manager: Optional DatabaseManager instance (creates one if None)
            enable_nlp: Enable NLP text analysis (only if NLP module is available)
        """
        self.db_manager = db_manager or DatabaseManager()

        # NLP
        self.enable_nlp = bool(enable_nlp) and NLP_AVAILABLE
        self.text_analyzer = None
        if self.enable_nlp and TextAnalyzer is not None:
            try:
                self.text_analyzer = TextAnalyzer()
                logger.info("NLP text analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NLP analyzer: {e}. Continuing without NLP.")
                self.enable_nlp = False
                self.text_analyzer = None

        # Credibility
        self.credibility_calculator = None
        if CREDIBILITY_AVAILABLE and CredibilityCalculator is not None:
            try:
                self.credibility_calculator = CredibilityCalculator(self.db_manager)
                logger.info("Credibility calculator initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize credibility calculator: {e}. Continuing without credibility updates."
                )
                self.credibility_calculator = None

        # Reputation
        self.reputation_calculator = None
        if REPUTATION_AVAILABLE and ReputationCalculator is not None:
            try:
                self.reputation_calculator = ReputationCalculator(self.db_manager, self.credibility_calculator)
                logger.info("Reputation calculator initialized")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize reputation calculator: {e}. Continuing without reputation updates."
                )
                self.reputation_calculator = None

        logger.info("ReportHandler initialized")

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def submit_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a new report.

        Flow:
        1) Validate payload -> Report model
        2) (Optional) NLP analysis
        3) Raw JSON backup (best-effort)
        4) Insert report into DB
        5) (Optional) update credibility + reputation (best-effort)
        """
        try:
            reporter_id = report_data.get("reporter_id", "unknown")
            merchant_id = report_data.get("merchant_id", "unknown")
            logger.info(f"Processing new report: reporter={reporter_id}, merchant={merchant_id}")

            report = self._validate_report(report_data)
            if report is None:
                return {"status": "error", "message": "Validation failed", "report_id": None}

            # Optional NLP analysis
            nlp_analysis: Optional[Dict[str, Any]] = None
            if self.enable_nlp and self.text_analyzer is not None:
                try:
                    nlp_analysis = self.text_analyzer.analyze_report(
                        title=report.title,
                        description=report.description,
                    )
                except Exception as e:
                    logger.warning(f"NLP analysis failed: {e}. Continuing without NLP results.")
                    nlp_analysis = None

            # Save raw backup (best-effort)
            self._save_raw_backup(report, nlp_analysis=nlp_analysis)

            # Prepare dict for DB
            report_dict = report.to_dict()

            # Attach NLP results into metadata_json for storage (if present)
            if nlp_analysis:
                try:
                    metadata_dict = {}
                    if report_dict.get("metadata_json"):
                        metadata_dict = json.loads(report_dict["metadata_json"])
                    metadata_dict["nlp_analysis"] = nlp_analysis
                    report_dict["metadata_json"] = json.dumps(metadata_dict)
                except Exception as e:
                    logger.warning(f"Failed to attach NLP analysis into metadata_json: {e}")

            # Insert into DB
            success = self.db_manager.insert_report(report_dict)

            # Optional credibility + reputation updates
            if success:
                payload_for_scoring = dict(report_dict)
                if nlp_analysis:
                    payload_for_scoring["nlp_analysis"] = nlp_analysis

                if self.credibility_calculator is not None:
                    try:
                        self.credibility_calculator.update_reporter_credibility(
                            reporter_id=report.reporter_id,
                            report_data=payload_for_scoring,
                        )
                    except Exception as e:
                        logger.warning(f"Credibility update failed: {e}")

                if self.reputation_calculator is not None:
                    try:
                        self.reputation_calculator.update_merchant_reputation(
                            merchant_id=report.merchant_id,
                            report_data=payload_for_scoring,
                        )
                    except Exception as e:
                        logger.warning(f"Reputation update failed: {e}")

                logger.info(f"Report {report.report_id} stored successfully")
                return {
                    "status": "success",
                    "message": "Report submitted successfully",
                    "report_id": report.report_id,
                    "timestamp": report.timestamp.isoformat(),
                }

            logger.error("Failed to store report in database")
            return {
                "status": "error",
                "message": "Failed to store report in database",
                "report_id": report.report_id,
            }

        except Exception as e:
            logger.error(f"Unexpected error during report submission: {e}")
            return {"status": "error", "message": "An internal error occurred", "report_id": None}

    def get_report(self, report_id: str) -> Optional[Report]:
        """Retrieve a report by ID."""
        try:
            report_data = self.db_manager.get_report_by_id(report_id)
            return Report.from_dict(report_data) if report_data else None
        except Exception as e:
            logger.error(f"Failed to retrieve report {report_id}: {e}")
            return None

    def get_merchant_reports(self, merchant_id: str, limit: int = 100) -> list:
        try:
            reports_data = self.db_manager.get_reports_by_merchant(merchant_id, limit)
            out = []
            for row in reports_data:
                try:
                    out.append(Report.from_dict(row))
                except Exception as e:
                    logger.warning(f"Skipping bad report row for merchant {merchant_id}: {e}")
            return out
        except Exception as e:
            logger.error(f"Failed to retrieve merchant reports: {e}")
            return []


    def get_reporter_reports(self, reporter_id: str, limit: int = 100) -> list:
        try:
            reports_data = self.db_manager.get_reports_by_reporter(reporter_id, limit)
            out = []
            for row in reports_data:
                try:
                    out.append(Report.from_dict(row))
                except Exception as e:
                    logger.warning(f"Skipping bad report row for reporter {reporter_id}: {e}")
            return out
        except Exception as e:
            logger.error(f"Failed to retrieve reporter reports: {e}")
            return []


    def get_statistics(self) -> Dict[str, Any]:
        """Get system stats from DB manager."""
        try:
            return self.db_manager.get_stats()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def analyze_report_text(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Run NLP analysis for a specific report (on-demand)."""
        if not self.enable_nlp or self.text_analyzer is None:
            return None

        report = self.get_report(report_id)
        if not report:
            return None

        try:
            return self.text_analyzer.analyze_report(title=report.title, description=report.description)
        except Exception as e:
            logger.error(f"Failed to analyze report text for {report_id}: {e}")
            return None

    def get_report_with_analysis(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Return report + on-demand NLP analysis (if available)."""
        report = self.get_report(report_id)
        if not report:
            return None

        result = report.model_dump()
        analysis = self.analyze_report_text(report_id)
        if analysis:
            result["nlp_analysis"] = analysis
        return result

    def get_reporter_credibility(self, reporter_id: str) -> Optional[Dict[str, Any]]:
        """Get credibility info for a reporter."""
        if self.credibility_calculator is None:
            return None
        try:
            cred = self.credibility_calculator.calculate_credibility(reporter_id)
            return _safe_dict(cred)
        except Exception as e:
            logger.error(f"Failed to get reporter credibility for {reporter_id}: {e}")
            return None

    def update_reporter_credibility(self, reporter_id: str) -> Optional[Dict[str, Any]]:
        """Manually trigger credibility update for a reporter."""
        if self.credibility_calculator is None:
            return None
        try:
            cred = self.credibility_calculator.update_reporter_credibility(reporter_id)
            return _safe_dict(cred)
        except Exception as e:
            logger.error(f"Failed to update reporter credibility for {reporter_id}: {e}")
            return None

    def get_merchant_reputation(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        """Get reputation info for a merchant."""
        if self.reputation_calculator is None:
            return None
        try:
            rep = self.reputation_calculator.calculate_reputation(merchant_id)
            return _safe_dict(rep)
        except Exception as e:
            logger.error(f"Failed to get merchant reputation for {merchant_id}: {e}")
            return None

    def update_merchant_reputation(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        """Manually trigger reputation update for a merchant."""
        if self.reputation_calculator is None:
            return None
        try:
            rep = self.reputation_calculator.update_merchant_reputation(merchant_id)
            return _safe_dict(rep)
        except Exception as e:
            logger.error(f"Failed to update merchant reputation for {merchant_id}: {e}")
            return None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _validate_report(self, report_data: Dict[str, Any]) -> Optional[Report]:
        """Validate incoming data and return Report or None."""
        try:
            data = dict(report_data)

            # Normalize metadata
            if "metadata" in data and isinstance(data["metadata"], dict):
                data["metadata"] = ReportMetadata(**data["metadata"])

            report = Report(**data)
            logger.info(f"Report validation OK: report_id={report.report_id}")
            return report

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            self._log_validation_errors(e)
            return None
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return None

    def _log_validation_errors(self, error: ValidationError) -> None:
        for err in error.errors():
            field = " -> ".join(str(loc) for loc in err.get("loc", []))
            msg = err.get("msg", "validation error")
            logger.error(f"Validation error in field '{field}': {msg}")

    def _resolve_raw_dir(self) -> Path:
        """
        Determine raw backup directory.
        Priority:
        1) RAW_DATA_DIR from settings if valid
        2) customer_reputation_system/data/raw
        """
        if RAW_DATA_DIR is not None:
            try:
                p = Path(RAW_DATA_DIR)
                p.mkdir(parents=True, exist_ok=True)
                return p
            except Exception:
                pass

        fallback = Path("customer_reputation_system") / "data" / "raw"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    def _save_raw_backup(self, report: Report, nlp_analysis: Optional[Dict[str, Any]] = None) -> None:
        """Best-effort save raw report JSON backup."""
        try:
            raw_dir = self._resolve_raw_dir()
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{report.report_id}_{timestamp_str}.json"
            path = raw_dir / filename

            payload = report.model_dump(mode="json")
            if nlp_analysis:
                payload["nlp_analysis"] = nlp_analysis

            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=_json_default)

            logger.debug(f"Raw backup saved: {path}")
        except Exception as e:
            logger.warning(f"Failed to save raw backup: {e}")
