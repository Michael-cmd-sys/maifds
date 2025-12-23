"""
Database operations for report storage
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path

from customer_reputation_system.config.settings import DATABASE_PATH
from customer_reputation_system.config.logging_config import setup_logger
from customer_reputation_system.src.storage.schemas import ALL_SCHEMAS

logger = setup_logger(__name__)


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self, db_path: Path = DATABASE_PATH):
        """Initialize database manager"""
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self):
        """Create tables and indexes if they don't exist"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                for schema in ALL_SCHEMAS:
                    cursor.execute(schema)
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn

    def insert_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Insert a report into the database

        Args:
            report_data: Dictionary containing report data

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Insert report
                cursor.execute(
                    """
                    INSERT INTO reports (
                        report_id, timestamp, reporter_id, merchant_id,
                        report_type, rating, title, description,
                        transaction_id, amount, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        report_data["report_id"],
                        report_data["timestamp"],
                        report_data["reporter_id"],
                        report_data["merchant_id"],
                        report_data["report_type"],
                        report_data.get("rating"),
                        report_data["title"],
                        report_data["description"],
                        report_data.get("transaction_id"),
                        report_data.get("amount"),
                        report_data.get("metadata_json"),
                    ),
                )

                # Update or create reporter entry
                self._upsert_reporter(cursor, report_data["reporter_id"])

                # Update or create merchant entry
                self._upsert_merchant(cursor, report_data["merchant_id"])

                conn.commit()
                logger.info(f"Report {report_data['report_id']} inserted successfully")
                return True

        except sqlite3.IntegrityError as e:
            logger.error(
                f"Report with ID {report_data['report_id']} already exists: {e}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to insert report: {e}")
            return False

    def _upsert_reporter(self, cursor: sqlite3.Cursor, reporter_id: str):
        """Update or insert reporter record"""
        cursor.execute(
            """
            INSERT INTO reporters (reporter_id, total_reports)
            VALUES (?, 1)
            ON CONFLICT(reporter_id) DO UPDATE SET
                total_reports = total_reports + 1,
                updated_at = CURRENT_TIMESTAMP
        """,
            (reporter_id,),
        )

    def _upsert_merchant(self, cursor: sqlite3.Cursor, merchant_id: str):
        """Update or insert merchant record"""
        cursor.execute(
            """
            INSERT INTO merchants (merchant_id, total_reports)
            VALUES (?, 1)
            ON CONFLICT(merchant_id) DO UPDATE SET
                total_reports = total_reports + 1,
                updated_at = CURRENT_TIMESTAMP
        """,
            (merchant_id,),
        )

    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a report by ID

        Args:
            report_id: Report identifier

        Returns:
            Report data as dictionary or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM reports WHERE report_id = ?", (report_id,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve report {report_id}: {e}")
            return None

    def get_reports_by_merchant(
        self, merchant_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all reports for a specific merchant

        Args:
            merchant_id: Merchant identifier
            limit: Maximum number of reports to return

        Returns:
            List of report dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM reports 
                    WHERE merchant_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (merchant_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve reports for merchant {merchant_id}: {e}")
            return []

    def get_reports_by_reporter(
        self, reporter_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all reports by a specific reporter

        Args:
            reporter_id: Reporter identifier
            limit: Maximum number of reports to return

        Returns:
            List of report dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM reports 
                    WHERE reporter_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (reporter_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve reports by reporter {reporter_id}: {e}")
            return []

    def get_all_reports(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all reports

        Args:
            limit: Maximum number of reports to return

        Returns:
            List of report dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM reports ORDER BY timestamp DESC LIMIT ?", (limit,)
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve reports: {e}")
            return []

    def delete_report(self, report_id: str) -> bool:
        """
        Delete a report by ID

        Args:
            report_id: Report identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))
                conn.commit()
                if cursor.rowcount > 0:
                    logger.info(f"Report {report_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"Report {report_id} not found")
                    return False
        except Exception as e:
            logger.error(f"Failed to delete report {report_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) as total FROM reports")
                total_reports = cursor.fetchone()["total"]

                cursor.execute("SELECT COUNT(*) as total FROM reporters")
                total_reporters = cursor.fetchone()["total"]

                cursor.execute("SELECT COUNT(*) as total FROM merchants")
                total_merchants = cursor.fetchone()["total"]

                return {
                    "total_reports": total_reports,
                    "total_reporters": total_reporters,
                    "total_merchants": total_merchants,
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
