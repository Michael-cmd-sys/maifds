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


class ReportHandler:
    """Handles report submission and ingestion"""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize report handler

        Args:
            db_manager: Database manager instance (creates new one if None)
        """
        self.db_manager = db_manager or DatabaseManager()
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
                "message": f"Internal error: {str(e)}",
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
