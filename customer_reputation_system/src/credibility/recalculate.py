"""
Script to recalculate credibility scores for all reporters
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.storage.database import DatabaseManager
from src.credibility.calculator import CredibilityCalculator
from config.logging_config import setup_logger

logger = setup_logger(__name__)


def recalculate_all_credibilities():
    """Recalculate credibility scores for all reporters"""
    db_manager = DatabaseManager()
    calculator = CredibilityCalculator(db_manager)

    # Get all reporters
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT reporter_id FROM reporters")
            reporters = [row["reporter_id"] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get reporters: {e}")
        return

    logger.info(f"Recalculating credibility for {len(reporters)} reporters...")

    updated = 0
    failed = 0

    for reporter_id in reporters:
        try:
            credibility = calculator.update_reporter_credibility(reporter_id)
            logger.info(
                f"Reporter {reporter_id}: credibility = {credibility.credibility_score:.3f}"
            )
            updated += 1
        except Exception as e:
            logger.error(f"Failed to update credibility for {reporter_id}: {e}")
            failed += 1

    logger.info(f"Credibility recalculation complete: {updated} updated, {failed} failed")


if __name__ == "__main__":
    recalculate_all_credibilities()

