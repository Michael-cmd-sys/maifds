"""
Merchant Reputation Calculator

Calculates merchant reputation scores based on:
- Credibility-weighted ratings
- Sentiment analysis from NLP
- Fraud report ratio
- Report volume
- Time-based factors
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import math

from src.reputation.models import MerchantReputation, ReputationFactors
from src.reputation.config import (
    RATING_WEIGHT,
    SENTIMENT_WEIGHT,
    FRAUD_RISK_WEIGHT,
    VOLUME_WEIGHT,
    TIME_DECAY_WEIGHT,
    MIN_RATING,
    MAX_RATING,
    RATING_TO_SCORE_DIVISOR,
    MIN_CREDIBILITY_FOR_WEIGHT,
    CREDIBILITY_BOOST_THRESHOLD,
    CREDIBILITY_BOOST_AMOUNT,
    POSITIVE_SENTIMENT_VALUE,
    NEUTRAL_SENTIMENT_VALUE,
    NEGATIVE_SENTIMENT_VALUE,
    FRAUD_REPORT_PENALTY,
    MAX_FRAUD_PENALTY,
    MIN_REPORTS_FOR_VOLUME,
    OPTIMAL_REPORT_COUNT,
    MAX_VOLUME_SCORE,
    RECENT_REPORT_DAYS,
    TIME_DECAY_HALF_LIFE_DAYS,
    TREND_WINDOW_DAYS,
    TREND_THRESHOLD,
    INITIAL_REPUTATION,
)
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.logging_config import setup_logger

logger = setup_logger(__name__)


class ReputationCalculator:
    """Calculates merchant reputation scores"""

    def __init__(self, db_manager, credibility_calculator=None):
        """
        Initialize reputation calculator

        Args:
            db_manager: DatabaseManager instance
            credibility_calculator: Optional CredibilityCalculator instance
        """
        self.db_manager = db_manager
        self.credibility_calculator = credibility_calculator

    def calculate_reputation(
        self, merchant_id: str, report_data: Optional[Dict[str, Any]] = None
    ) -> MerchantReputation:
        """
        Calculate reputation score for a merchant

        Args:
            merchant_id: Merchant identifier
            report_data: Optional new report data (for incremental updates)

        Returns:
            MerchantReputation object
        """
        # Get all reports for this merchant
        reports = self._get_merchant_reports(merchant_id)

        if not reports:
            # New merchant - return initial reputation
            return MerchantReputation(
                merchant_id=merchant_id,
                reputation_score=INITIAL_REPUTATION,
                total_reports=0,
            )

        # Calculate factors
        factors = self._calculate_factors(merchant_id, reports, report_data)

        # Calculate overall reputation score
        reputation_score = self._combine_factors(factors, merchant_id, reports)

        # Calculate additional metrics
        metrics = self._calculate_metrics(reports)

        # Calculate trend
        trend = self._calculate_trend(reports)

        return MerchantReputation(
            merchant_id=merchant_id,
            reputation_score=reputation_score,
            total_reports=len(reports),
            average_rating=metrics.get("average_rating"),
            credibility_weighted_rating=metrics.get("credibility_weighted_rating"),
            positive_reports_ratio=metrics.get("positive_reports_ratio"),
            negative_reports_ratio=metrics.get("negative_reports_ratio"),
            fraud_reports_ratio=metrics.get("fraud_reports_ratio"),
            recent_trend=trend,
            updated_at=datetime.utcnow(),
        )

    def _get_merchant_reports(self, merchant_id: str) -> List[Dict[str, Any]]:
        """Get all reports for a merchant"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM reports 
                    WHERE merchant_id = ? 
                    ORDER BY timestamp DESC
                """,
                    (merchant_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get merchant reports: {e}")
            return []

    def _get_reporter_credibility(self, reporter_id: str) -> float:
        """Get reporter credibility score"""
        if not self.credibility_calculator:
            # Fallback to database
            try:
                with self.db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT credibility_score FROM reporters WHERE reporter_id = ?",
                        (reporter_id,),
                    )
                    row = cursor.fetchone()
                    if row and row["credibility_score"] is not None:
                        return float(row["credibility_score"])
            except Exception:
                pass
            return INITIAL_REPUTATION

        try:
            credibility = self.credibility_calculator.calculate_credibility(reporter_id)
            return credibility.credibility_score
        except Exception as e:
            logger.warning(f"Failed to get credibility for {reporter_id}: {e}")
            return INITIAL_REPUTATION

    def _calculate_factors(
        self,
        merchant_id: str,
        reports: List[Dict[str, Any]],
        new_report: Optional[Dict[str, Any]] = None,
    ) -> ReputationFactors:
        """Calculate all reputation factors"""

        # 1. Credibility-weighted rating
        weighted_rating = self._calculate_weighted_rating(reports, new_report)

        # 2. Sentiment score (from NLP)
        sentiment_score = self._calculate_sentiment_score(reports, new_report)

        # 3. Fraud risk (inverse - lower fraud = higher score)
        fraud_risk = self._calculate_fraud_risk(reports)

        # 4. Report volume score
        volume_score = self._calculate_volume_score(reports)

        # 5. Time decay (recent activity)
        time_decay = self._calculate_time_decay(reports)

        # 6. Average reporter credibility
        avg_credibility = self._calculate_average_credibility(reports)

        return ReputationFactors(
            weighted_rating_score=weighted_rating,
            sentiment_score=sentiment_score,
            fraud_risk_score=1.0 - fraud_risk,  # Inverse: lower fraud = higher score
            report_volume_score=volume_score,
            time_decay_factor=time_decay,
            credibility_weight=avg_credibility,
        )

    def _calculate_weighted_rating(
        self, reports: List[Dict[str, Any]], new_report: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate credibility-weighted average rating"""
        weighted_sum = 0.0
        total_weight = 0.0

        # Process existing reports
        for report in reports:
            rating = report.get("rating")
            if rating is None:
                continue

            reporter_id = report.get("reporter_id")
            credibility = self._get_reporter_credibility(reporter_id)

            # Only count if credibility meets minimum threshold
            if credibility >= MIN_CREDIBILITY_FOR_WEIGHT:
                weight = credibility
                # Boost for high credibility reporters
                if credibility >= CREDIBILITY_BOOST_THRESHOLD:
                    weight += CREDIBILITY_BOOST_AMOUNT

                weighted_sum += rating * weight
                total_weight += weight

        # Process new report if provided
        if new_report and new_report.get("rating") is not None:
            reporter_id = new_report.get("reporter_id")
            if reporter_id:
                credibility = self._get_reporter_credibility(reporter_id)
                if credibility >= MIN_CREDIBILITY_FOR_WEIGHT:
                    weight = credibility
                    if credibility >= CREDIBILITY_BOOST_THRESHOLD:
                        weight += CREDIBILITY_BOOST_AMOUNT
                    weighted_sum += new_report["rating"] * weight
                    total_weight += weight

        if total_weight == 0:
            # No valid ratings - return neutral
            return INITIAL_REPUTATION

        avg_rating = weighted_sum / total_weight
        # Normalize to 0-1
        return (avg_rating - MIN_RATING) / (MAX_RATING - MIN_RATING)

    def _calculate_sentiment_score(
        self, reports: List[Dict[str, Any]], new_report: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate sentiment score from NLP analysis"""
        sentiment_values = []

        # Process existing reports
        for report in reports:
            metadata_json = report.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    nlp_analysis = metadata.get("nlp_analysis", {})
                    sentiment = nlp_analysis.get("sentiment")
                    if sentiment:
                        if sentiment == "positive":
                            sentiment_values.append(POSITIVE_SENTIMENT_VALUE)
                        elif sentiment == "neutral":
                            sentiment_values.append(NEUTRAL_SENTIMENT_VALUE)
                        elif sentiment == "negative":
                            sentiment_values.append(NEGATIVE_SENTIMENT_VALUE)
                except (json.JSONDecodeError, (ValueError, TypeError)):
                    continue

        # Process new report if provided
        if new_report:
            nlp_analysis = new_report.get("nlp_analysis", {})
            sentiment = nlp_analysis.get("sentiment")
            if sentiment:
                if sentiment == "positive":
                    sentiment_values.append(POSITIVE_SENTIMENT_VALUE)
                elif sentiment == "neutral":
                    sentiment_values.append(NEUTRAL_SENTIMENT_VALUE)
                elif sentiment == "negative":
                    sentiment_values.append(NEGATIVE_SENTIMENT_VALUE)

        if not sentiment_values:
            # No NLP analysis - use rating as proxy
            ratings = [r.get("rating") for r in reports if r.get("rating")]
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                # Convert rating to sentiment (1-2 = negative, 3 = neutral, 4-5 = positive)
                if avg_rating <= 2:
                    return NEGATIVE_SENTIMENT_VALUE
                elif avg_rating >= 4:
                    return POSITIVE_SENTIMENT_VALUE
                else:
                    return NEUTRAL_SENTIMENT_VALUE
            return INITIAL_REPUTATION

        return sum(sentiment_values) / len(sentiment_values)

    def _calculate_fraud_risk(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate fraud risk score (0-1, higher = more fraud)"""
        if not reports:
            return 0.0

        fraud_count = sum(
            1 for r in reports if r.get("report_type") == "fraud"
        )
        fraud_ratio = fraud_count / len(reports)

        # Apply penalty scaling
        penalty = min(MAX_FRAUD_PENALTY, fraud_ratio * FRAUD_REPORT_PENALTY * len(reports))

        return min(1.0, fraud_ratio + penalty)

    def _calculate_volume_score(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate report volume score"""
        count = len(reports)

        if count < MIN_REPORTS_FOR_VOLUME:
            # Too few reports - low confidence
            return count / MIN_REPORTS_FOR_VOLUME * 0.5

        # Optimal volume around OPTIMAL_REPORT_COUNT
        if count <= OPTIMAL_REPORT_COUNT:
            return min(MAX_VOLUME_SCORE, count / OPTIMAL_REPORT_COUNT)
        else:
            # Diminishing returns after optimal
            excess = count - OPTIMAL_REPORT_COUNT
            return max(0.8, MAX_VOLUME_SCORE - (excess / OPTIMAL_REPORT_COUNT * 0.2))

    def _calculate_time_decay(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate time decay factor (recent activity boost)"""
        if not reports:
            return 0.0

        now = datetime.utcnow()
        recent_count = 0
        total_weight = 0.0

        for report in reports:
            try:
                if isinstance(report.get("timestamp"), str):
                    report_time = datetime.fromisoformat(report["timestamp"])
                else:
                    report_time = report.get("timestamp", now)

                days_ago = (now - report_time).days

                # Exponential decay
                weight = math.exp(-days_ago / TIME_DECAY_HALF_LIFE_DAYS)
                total_weight += weight

                if days_ago <= RECENT_REPORT_DAYS:
                    recent_count += 1

            except (ValueError, TypeError):
                continue

        if len(reports) > 0:
            avg_weight = total_weight / len(reports)
            recent_ratio = recent_count / len(reports)
            return (avg_weight + recent_ratio) / 2.0

        return 0.0

    def _calculate_average_credibility(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate average reporter credibility"""
        if not reports:
            return INITIAL_REPUTATION

        credibilities = []
        for report in reports:
            reporter_id = report.get("reporter_id")
            if reporter_id:
                cred = self._get_reporter_credibility(reporter_id)
                credibilities.append(cred)

        if not credibilities:
            return INITIAL_REPUTATION

        return sum(credibilities) / len(credibilities)

    def _calculate_metrics(self, reports: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate additional reputation metrics"""
        metrics = {}

        # Average rating
        ratings = [r.get("rating") for r in reports if r.get("rating")]
        if ratings:
            metrics["average_rating"] = sum(ratings) / len(ratings)

        # Credibility-weighted rating
        weighted_sum = 0.0
        total_weight = 0.0
        for report in reports:
            rating = report.get("rating")
            if rating is None:
                continue
            reporter_id = report.get("reporter_id")
            credibility = self._get_reporter_credibility(reporter_id)
            if credibility >= MIN_CREDIBILITY_FOR_WEIGHT:
                weight = credibility
                if credibility >= CREDIBILITY_BOOST_THRESHOLD:
                    weight += CREDIBILITY_BOOST_AMOUNT
                weighted_sum += rating * weight
                total_weight += weight
        if total_weight > 0:
            metrics["credibility_weighted_rating"] = weighted_sum / total_weight

        # Sentiment ratios
        positive = 0
        negative = 0
        fraud = 0
        total_with_sentiment = 0

        for report in reports:
            metadata_json = report.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    nlp_analysis = metadata.get("nlp_analysis", {})
                    sentiment = nlp_analysis.get("sentiment")
                    if sentiment:
                        total_with_sentiment += 1
                        if sentiment == "positive":
                            positive += 1
                        elif sentiment == "negative":
                            negative += 1
                except (json.JSONDecodeError, (ValueError, TypeError)):
                    pass

            if report.get("report_type") == "fraud":
                fraud += 1

        if total_with_sentiment > 0:
            metrics["positive_reports_ratio"] = positive / total_with_sentiment
            metrics["negative_reports_ratio"] = negative / total_with_sentiment

        if reports:
            metrics["fraud_reports_ratio"] = fraud / len(reports)

        return metrics

    def _calculate_trend(self, reports: List[Dict[str, Any]]) -> Optional[str]:
        """Calculate recent trend (trending_up, trending_down, stable)"""
        if len(reports) < 2:
            return None

        now = datetime.utcnow()
        cutoff_date = now - timedelta(days=TREND_WINDOW_DAYS)

        recent_reports = []
        older_reports = []

        for report in reports:
            try:
                if isinstance(report.get("timestamp"), str):
                    report_time = datetime.fromisoformat(report["timestamp"])
                else:
                    report_time = report.get("timestamp", now)

                if report_time >= cutoff_date:
                    recent_reports.append(report)
                else:
                    older_reports.append(report)
            except (ValueError, TypeError):
                continue

        if not recent_reports or not older_reports:
            return None

        # Calculate average ratings
        recent_ratings = [r.get("rating") for r in recent_reports if r.get("rating")]
        older_ratings = [r.get("rating") for r in older_reports if r.get("rating")]

        if not recent_ratings or not older_ratings:
            return None

        recent_avg = sum(recent_ratings) / len(recent_ratings)
        older_avg = sum(older_ratings) / len(older_ratings)

        diff = recent_avg - older_avg

        if abs(diff) < TREND_THRESHOLD:
            return "stable"
        elif diff > 0:
            return "trending_up"
        else:
            return "trending_down"

    def _combine_factors(
        self,
        factors: ReputationFactors,
        merchant_id: str,
        reports: List[Dict[str, Any]],
    ) -> float:
        """Combine all factors into final reputation score"""

        # Base score from factors
        base_score = (
            factors.weighted_rating_score * RATING_WEIGHT
            + factors.sentiment_score * SENTIMENT_WEIGHT
            + factors.fraud_risk_score * FRAUD_RISK_WEIGHT
            + factors.report_volume_score * VOLUME_WEIGHT
            + factors.time_decay_factor * TIME_DECAY_WEIGHT
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_score))

    def update_merchant_reputation(
        self, merchant_id: str, report_data: Optional[Dict[str, Any]] = None
    ) -> MerchantReputation:
        """
        Calculate and update merchant reputation in database

        Args:
            merchant_id: Merchant identifier
            report_data: Optional new report data

        Returns:
            Updated MerchantReputation object
        """
        reputation = self.calculate_reputation(merchant_id, report_data)

        # Update database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE merchants 
                    SET reputation_score = ?,
                        average_rating = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE merchant_id = ?
                """,
                    (
                        reputation.reputation_score,
                        reputation.average_rating,
                        merchant_id,
                    ),
                )
                conn.commit()
                logger.info(
                    f"Updated reputation for merchant {merchant_id}: {reputation.reputation_score:.3f}"
                )
        except Exception as e:
            logger.error(f"Failed to update merchant reputation: {e}")

        return reputation

