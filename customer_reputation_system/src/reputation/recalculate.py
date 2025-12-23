"""
Script to recalculate reputation scores for all merchants
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from customer_reputation_system.src.storage.database import DatabaseManager
from customer_reputation_system.src.credibility.calculator import CredibilityCalculator
from customer_reputation_system.src.reputation.calculator import ReputationCalculator
from customer_reputation_system.config.logging_config import setup_logger

logger = setup_logger(__name__)


def recalculate_all_reputations():
    """Recalculate reputation scores for all merchants"""
    db_manager = DatabaseManager()
    credibility_calculator = CredibilityCalculator(db_manager)
    reputation_calculator = ReputationCalculator(db_manager, credibility_calculator)

    # Get all merchants
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT merchant_id FROM merchants")
            merchants = [row["merchant_id"] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get merchants: {e}")
        return

    logger.info(f"Recalculating reputation for {len(merchants)} merchants...")

    updated = 0
    failed = 0

    for merchant_id in merchants:
        try:
            reputation = reputation_calculator.update_merchant_reputation(merchant_id)
            logger.info(
                f"Merchant {merchant_id}: reputation = {reputation.reputation_score:.3f}, "
                f"rating = {reputation.average_rating:.2f if reputation.average_rating else 'N/A'}"
            )
            updated += 1
        except Exception as e:
            logger.error(f"Failed to update reputation for {merchant_id}: {e}")
            failed += 1

    logger.info(f"Reputation recalculation complete: {updated} updated, {failed} failed")


if __name__ == "__main__":
    recalculate_all_reputations()

