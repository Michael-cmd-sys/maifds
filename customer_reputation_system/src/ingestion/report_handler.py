"""
Main report ingestion handler
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime

from pydantic import ValidationError

from src.models.report_model import Report, ReportMetadata
from src.storage.database import DatabaseManager
from config.logging_config import setup_logger
from config.settings import RAW_DATA_DIR

logger = setup_logger(__name__)

# Optional NLP import (graceful degradation if not available)
try:
    from src.nlp.text_analyzer import TextAnalyzer
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP module not available. Text analysis will be skipped.")

# Credibility system import
try:
    from src.credibility.calculator import CredibilityCalculator
    CREDIBILITY_AVAILABLE = True
except ImportError:
    CREDIBILITY_AVAILABLE = False
    logger.warning("Credibility module not available. Credibility scoring will be skipped.")

# Reputation system import
try:
    from src.reputation.calculator import ReputationCalculator
    REPUTATION_AVAILABLE = True
except ImportError:
    REPUTATION_AVAILABLE = False
    logger.warning("Reputation module not available. Reputation scoring will be skipped.")


class ReportHandler:
    """Handles report submission and ingestion"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
    def __init__(self, db_manager: Optional[DatabaseManager] = None, enable_nlp: bool = True):
        """
        Initialize report handler

        Args:
            db_manager: Database manager instance (creates new one if None)
        """
        self.db_manager = db_manager or DatabaseManager()
            enable_nlp: Enable NLP text analysis (default: True)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.enable_nlp = enable_nlp and NLP_AVAILABLE
        
        # Initialize NLP analyzer if available
        self.text_analyzer = None
        if self.enable_nlp:
            try:
                self.text_analyzer = TextAnalyzer()
                logger.info("NLP text analyzer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NLP analyzer: {e}. Continuing without NLP.")
                self.enable_nlp = False

        # Initialize credibility calculator if available
        self.credibility_calculator = None
        if CREDIBILITY_AVAILABLE:
            try:
                self.credibility_calculator = CredibilityCalculator(self.db_manager)
                logger.info("Credibility calculator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize credibility calculator: {e}. Continuing without credibility updates.")

        # Initialize reputation calculator if available
        self.reputation_calculator = None
        if REPUTATION_AVAILABLE:
            try:
                self.reputation_calculator = ReputationCalculator(
                    self.db_manager, self.credibility_calculator
                )
                logger.info("Reputation calculator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize reputation calculator: {e}. Continuing without reputation updates.")
        
        logger.info("ReportHandler initialized")

    def submit_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a new report

        Args:
            report_data: Dictionary containing report information

        Returns:
            Dictionary with status and details
        """
        try:
            # Step 1: Validate and create Report model
            logger.info(
                f"Processing new report from reporter: {report_data.get('reporter_id')}"
            )
            report = self._validate_report(report_data)

            if not report:
                return {
                    "status": "error",
                    "message": "Validation failed",
                    "report_id": None,
                }

            # Step 2: Save to backup (raw JSON)
            self._save_raw_backup(report)

            # Step 3: Store in database
            success = self.db_manager.insert_report(report.to_dict())
            # Step 2: Perform NLP analysis (if enabled)
            nlp_analysis = None
            if self.enable_nlp and self.text_analyzer:
                try:
                    nlp_analysis = self.text_analyzer.analyze_report(
                        title=report.title,
                        description=report.description
                    )
                    logger.debug(f"NLP analysis completed for report {report.report_id}")
                except Exception as e:
                    logger.warning(f"NLP analysis failed: {e}. Continuing without NLP results.")

            # Step 3: Save to backup (raw JSON)
            self._save_raw_backup(report)

            # Step 4: Store in database (with NLP analysis if available)
            report_dict = report.to_dict()
            if nlp_analysis:
                # Add NLP results to metadata JSON
                import json
                metadata_dict = json.loads(report_dict.get("metadata_json", "{}")) if report_dict.get("metadata_json") else {}
                metadata_dict["nlp_analysis"] = nlp_analysis
                report_dict["metadata_json"] = json.dumps(metadata_dict)

            # Prepare report data with NLP for credibility calculation
            report_data_for_credibility = report_dict.copy()
            if nlp_analysis:
                report_data_for_credibility["nlp_analysis"] = nlp_analysis
            
            success = self.db_manager.insert_report(report_dict)

            # Step 5: Update reporter credibility (if enabled)
            if success and self.credibility_calculator:
                try:
                    self.credibility_calculator.update_reporter_credibility(
                        reporter_id=report.reporter_id,
                        report_data=report_data_for_credibility
                    )
                    logger.debug(f"Updated credibility for reporter {report.reporter_id}")
                except Exception as e:
                    logger.warning(f"Failed to update reporter credibility: {e}")
                    # Don't fail the report submission if credibility update fails

            # Step 6: Update merchant reputation (if enabled)
            if success and self.reputation_calculator:
                try:
                    self.reputation_calculator.update_merchant_reputation(
                        merchant_id=report.merchant_id,
                        report_data=report_data_for_credibility
                    )
                    logger.debug(f"Updated reputation for merchant {report.merchant_id}")
                except Exception as e:
                    logger.warning(f"Failed to update merchant reputation: {e}")
                    # Don't fail the report submission if reputation update fails

            if success:
                logger.info(f"Report {report.report_id} processed successfully")
                return {
                    "status": "success",
                    "message": "Report submitted successfully",
                    "report_id": report.report_id,
                    "timestamp": report.timestamp.isoformat(),
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to store report in database",
                    "report_id": report.report_id,
                }

        except Exception as e:
            logger.error(f"Unexpected error during report submission: {e}")
            return {
                "status": "error",
                "message": "An internal error occurred",
                "report_id": None,
            }

    def _validate_report(self, report_data: Dict[str, Any]) -> Optional[Report]:
        """
        Validate report data using Pydantic model

        Args:
            report_data: Raw report data

        Returns:
            Validated Report object or None if validation fails
        """
        try:
            # Handle metadata if present
            if "metadata" in report_data and isinstance(report_data["metadata"], dict):
                report_data["metadata"] = ReportMetadata(**report_data["metadata"])

            report = Report(**report_data)
            logger.info(f"Report validation successful: {report.report_id}")
            return report

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            self._log_validation_errors(e)
            return None
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return None

    def _log_validation_errors(self, error: ValidationError):
        """Log detailed validation errors"""
        for err in error.errors():
            field = " -> ".join(str(loc) for loc in err["loc"])
            logger.error(f"Validation error in field '{field}': {err['msg']}")

    def _save_raw_backup(self, report: Report):
        """
        Save raw report as JSON backup

        Args:
            report: Validated Report object
        """
        try:
            # Create filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{report.report_id}_{timestamp_str}.json"
            filepath = RAW_DATA_DIR / filename

            # Save as JSON
            with open(filepath, "w") as f:
                json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

            logger.debug(f"Raw backup saved: {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save raw backup: {e}")
            # Don't fail the whole operation if backup fails

    def get_report(self, report_id: str) -> Optional[Report]:
        """
        Retrieve a report by ID

        Args:
            report_id: Report identifier

        Returns:
            Report object or None if not found
        """
        try:
            report_data = self.db_manager.get_report_by_id(report_id)
            if report_data:
                return Report.from_dict(report_data)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve report {report_id}: {e}")
            return None

    def get_merchant_reports(self, merchant_id: str, limit: int = 100) -> list:
        """
        Get all reports for a merchant

        Args:
            merchant_id: Merchant identifier
            limit: Maximum number of reports

        Returns:
            List of Report objects
        """
        try:
            reports_data = self.db_manager.get_reports_by_merchant(merchant_id, limit)
            return [Report.from_dict(data) for data in reports_data]
        except Exception as e:
            logger.error(f"Failed to retrieve merchant reports: {e}")
            return []

    def get_reporter_reports(self, reporter_id: str, limit: int = 100) -> list:
        """
        Get all reports by a reporter

        Args:
            reporter_id: Reporter identifier
            limit: Maximum number of reports

        Returns:
            List of Report objects
        """
        try:
            reports_data = self.db_manager.get_reports_by_reporter(reporter_id, limit)
            return [Report.from_dict(data) for data in reports_data]
        except Exception as e:
            logger.error(f"Failed to retrieve reporter reports: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics

        Returns:
            Dictionary with statistics
        """
        return self.db_manager.get_stats()

    def analyze_report_text(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get NLP analysis for a specific report

        Args:
            report_id: Report identifier

        Returns:
            NLP analysis dictionary or None if not available
        """
        if not self.enable_nlp or not self.text_analyzer:
            logger.warning("NLP analysis not available")
            return None

        report = self.get_report(report_id)
        if not report:
            return None

        try:
            analysis = self.text_analyzer.analyze_report(
                title=report.title,
                description=report.description
            )
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze report text: {e}")
            return None

    def get_report_with_analysis(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Get report with NLP analysis included

        Args:
            report_id: Report identifier

        Returns:
            Dictionary with report data and NLP analysis, or None if not found
        """
        report = self.get_report(report_id)
        if not report:
            return None

        result = report.model_dump()
        nlp_analysis = self.analyze_report_text(report_id)
        
        if nlp_analysis:
            result["nlp_analysis"] = nlp_analysis

        return result

    def get_reporter_credibility(self, reporter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get reporter credibility information

        Args:
            reporter_id: Reporter identifier

        Returns:
            Dictionary with credibility information or None if not found
        """
        if not self.credibility_calculator:
            logger.warning("Credibility calculator not available")
            return None

        try:
            credibility = self.credibility_calculator.calculate_credibility(reporter_id)
            return credibility.to_dict()
        except Exception as e:
            logger.error(f"Failed to get reporter credibility: {e}")
            return None

    def update_reporter_credibility(self, reporter_id: str) -> Optional[Dict[str, Any]]:
        """
        Manually trigger credibility update for a reporter

        Args:
            reporter_id: Reporter identifier

        Returns:
            Updated credibility information or None if failed
        """
        if not self.credibility_calculator:
            logger.warning("Credibility calculator not available")
            return None

        try:
            credibility = self.credibility_calculator.update_reporter_credibility(reporter_id)
            return credibility.to_dict()
        except Exception as e:
            logger.error(f"Failed to update reporter credibility: {e}")
            return None

    def get_merchant_reputation(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get merchant reputation information

        Args:
            merchant_id: Merchant identifier

        Returns:
            Dictionary with reputation information or None if not found
        """
        if not self.reputation_calculator:
            logger.warning("Reputation calculator not available")
            return None

        try:
            reputation = self.reputation_calculator.calculate_reputation(merchant_id)
            return reputation.to_dict()
        except Exception as e:
            logger.error(f"Failed to get merchant reputation: {e}")
            return None

    def update_merchant_reputation(self, merchant_id: str) -> Optional[Dict[str, Any]]:
        """
        Manually trigger reputation update for a merchant

        Args:
            merchant_id: Merchant identifier

        Returns:
            Updated reputation information or None if failed
        """
        if not self.reputation_calculator:
            logger.warning("Reputation calculator not available")
            return None

        try:
            reputation = self.reputation_calculator.update_merchant_reputation(merchant_id)
            return reputation.to_dict()
        except Exception as e:
            logger.error(f"Failed to update merchant reputation: {e}")
            return None
