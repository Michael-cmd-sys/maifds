"""
Reporter Credibility Calculator

Calculates and updates reporter credibility scores based on:
- NLP text credibility scores
- Report consistency over time
- Verification status
- Time-based factors (recent activity)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import math

from customer_reputation_system.src.credibility.models import ReporterCredibility, CredibilityFactors
from customer_reputation_system.src.credibility.config import (
    TEXT_CREDIBILITY_WEIGHT,
    CONSISTENCY_WEIGHT,
    VERIFICATION_WEIGHT,
    TIME_DECAY_WEIGHT,
    RECENT_REPORT_DAYS,
    TIME_DECAY_HALF_LIFE_DAYS,
    MIN_REPORTS_FOR_CONSISTENCY,
    INITIAL_CREDIBILITY,
    VERIFICATION_BOOST,
    MIN_VERIFIED_FOR_BOOST,
)
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from customer_reputation_system.config.logging_config import setup_logger

logger = setup_logger(__name__)


class CredibilityCalculator:
    """Calculates reporter credibility scores"""

    def __init__(self, db_manager):
        """
        Initialize credibility calculator

        Args:
            db_manager: DatabaseManager instance
        """
        self.db_manager = db_manager

    def calculate_credibility(
        self, reporter_id: str, report_data: Optional[Dict[str, Any]] = None
    ) -> ReporterCredibility:
        """
        Calculate credibility score for a reporter

        Args:
            reporter_id: Reporter identifier
            report_data: Optional new report data (for incremental updates)

        Returns:
            ReporterCredibility object
        """
        # Get all reports for this reporter
        reports = self._get_reporter_reports(reporter_id)

        if not reports:
            # New reporter - return initial credibility
            return ReporterCredibility(
                reporter_id=reporter_id,
                credibility_score=INITIAL_CREDIBILITY,
                total_reports=0,
                verified_reports=0,
                average_text_credibility=None,
                consistency_score=None,
                recent_activity_score=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        # Calculate factors
        factors = self._calculate_factors(reporter_id, reports, report_data)

        # Calculate overall credibility score
        credibility_score = self._combine_factors(factors, reporter_id, reports)

        # Get reporter stats
        reporter_stats = self._get_reporter_stats(reporter_id)

        return ReporterCredibility(
            reporter_id=reporter_id,
            credibility_score=credibility_score,
            total_reports=reporter_stats.get("total_reports", len(reports)),
            verified_reports=reporter_stats.get("verified_reports", 0),
            average_text_credibility=factors.text_credibility_score,
            consistency_score=factors.report_consistency,
            recent_activity_score=factors.time_decay_factor,
            updated_at=datetime.utcnow(),
        )

    def _get_reporter_reports(self, reporter_id: str) -> List[Dict[str, Any]]:
        """Get all reports for a reporter"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM reports 
                    WHERE reporter_id = ? 
                    ORDER BY timestamp DESC
                """,
                    (reporter_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get reporter reports: {e}")
            return []

    def _get_reporter_stats(self, reporter_id: str) -> Dict[str, Any]:
        """Get reporter statistics from database"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM reporters WHERE reporter_id = ?", (reporter_id,)
                )
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return {}
        except Exception as e:
            logger.error(f"Failed to get reporter stats: {e}")
            return {}

    def _calculate_factors(
        self,
        reporter_id: str,
        reports: List[Dict[str, Any]],
        new_report: Optional[Dict[str, Any]] = None,
    ) -> CredibilityFactors:
        """Calculate all credibility factors"""

        # 1. Text credibility (from NLP analysis)
        text_credibility = self._calculate_text_credibility(reports, new_report)

        # 2. Report consistency
        consistency = self._calculate_consistency(reports)

        # 3. Verification rate
        verification_rate = self._calculate_verification_rate(reports)

        # 4. Time decay (recent activity)
        time_decay = self._calculate_time_decay(reports)

        # 5. Additional quality metrics
        quality_metrics = self._calculate_quality_metrics(reports)

        return CredibilityFactors(
            text_credibility_score=text_credibility,
            report_consistency=consistency,
            verification_rate=verification_rate,
            time_decay_factor=time_decay,
            report_quality_metrics=quality_metrics,
        )

    def _calculate_text_credibility(
        self, reports: List[Dict[str, Any]], new_report: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate average text credibility from NLP analysis"""
        credibility_scores = []

        # Process existing reports
        for report in reports:
            metadata_json = report.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                    nlp_analysis = metadata.get("nlp_analysis", {})
                    cred_score = nlp_analysis.get("credibility_score")
                    if cred_score is not None:
                        credibility_scores.append(float(cred_score))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue

        # Process new report if provided
        if new_report:
            nlp_analysis = new_report.get("nlp_analysis", {})
            cred_score = nlp_analysis.get("credibility_score")
            if cred_score is not None:
                credibility_scores.append(float(cred_score))

        if not credibility_scores:
            # No NLP analysis available - use default
            return INITIAL_CREDIBILITY

        return sum(credibility_scores) / len(credibility_scores)

    def _calculate_consistency(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate report consistency score"""
        if len(reports) < MIN_REPORTS_FOR_CONSISTENCY:
            # Not enough reports for consistency calculation
            return INITIAL_CREDIBILITY

        # Group reports by report_type
        type_counts = {}
        for report in reports:
            report_type = report.get("report_type", "other")
            type_counts[report_type] = type_counts.get(report_type, 0) + 1

        # Calculate consistency as similarity of report types
        # More consistent = higher score
        total_reports = len(reports)
        max_type_count = max(type_counts.values()) if type_counts else 0
        consistency = max_type_count / total_reports

        # Also consider rating consistency
        ratings = []
        for r in reports:
            rating = r.get("rating")
            if rating is not None:
                try:
                    ratings.append(float(rating))
                except (ValueError, TypeError):
                    continue
        
        if len(ratings) >= 2:
            rating_variance = self._calculate_variance(ratings)
            # Lower variance = higher consistency
            rating_consistency = 1.0 / (1.0 + rating_variance)
            consistency = (consistency + rating_consistency) / 2.0

        return min(1.0, consistency)

    def _calculate_verification_rate(self, reports: List[Dict[str, Any]]) -> float:
        """Calculate verification rate"""
        # For now, we use verified_reports from database
        # In future, this could be based on actual verification status
        reporter_stats = self._get_reporter_stats(reports[0]["reporter_id"])
        total = reporter_stats.get("total_reports", len(reports))
        verified = reporter_stats.get("verified_reports", 0)

        if total == 0:
            return 0.0

        return min(1.0, verified / total)

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

        # Normalize weight
        if len(reports) > 0:
            avg_weight = total_weight / len(reports)
            recent_ratio = recent_count / len(reports)
            return (avg_weight + recent_ratio) / 2.0

        return 0.0

    def _calculate_quality_metrics(
        self, reports: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate additional quality metrics"""
        metrics = {}

        # Average description length
        desc_lengths = [
            len(r.get("description", "")) for r in reports if r.get("description")
        ]
        if desc_lengths:
            metrics["avg_description_length"] = sum(desc_lengths) / len(desc_lengths)

        # Report completeness (has rating, transaction_id, etc.)
        completeness_scores = []
        for report in reports:
            score = 0.0
            if report.get("rating"):
                score += 0.3
            if report.get("transaction_id"):
                score += 0.3
            if report.get("amount"):
                score += 0.2
            if report.get("metadata_json"):
                score += 0.2
            completeness_scores.append(score)
        if completeness_scores:
            metrics["avg_completeness"] = sum(completeness_scores) / len(
                completeness_scores
            )

        return metrics

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _combine_factors(
        self,
        factors: CredibilityFactors,
        reporter_id: str,
        reports: List[Dict[str, Any]],
    ) -> float:
        """Combine all factors into final credibility score"""

        # Base score from factors
        base_score = (
            factors.text_credibility_score * TEXT_CREDIBILITY_WEIGHT
            + factors.report_consistency * CONSISTENCY_WEIGHT
            + factors.verification_rate * VERIFICATION_WEIGHT
            + factors.time_decay_factor * TIME_DECAY_WEIGHT
        )

        # Apply verification boost
        reporter_stats = self._get_reporter_stats(reporter_id)
        verified = reporter_stats.get("verified_reports", 0)
        if verified >= MIN_VERIFIED_FOR_BOOST:
            base_score += VERIFICATION_BOOST

        # Apply quality metrics boost
        quality_boost = factors.report_quality_metrics.get("avg_completeness", 0.0) * 0.1
        base_score += quality_boost

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_score))

    def update_reporter_credibility(
        self, reporter_id: str, report_data: Optional[Dict[str, Any]] = None
    ) -> ReporterCredibility:
        """
        Calculate and update reporter credibility in database

        Args:
            reporter_id: Reporter identifier
            report_data: Optional new report data

        Returns:
            Updated ReporterCredibility object
        """
        credibility = self.calculate_credibility(reporter_id, report_data)

        # Update database
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE reporters 
                    SET credibility_score = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE reporter_id = ?
                """,
                    (credibility.credibility_score, reporter_id),
                )
                conn.commit()
                logger.info(
                    f"Updated credibility for reporter {reporter_id}: {credibility.credibility_score:.3f}"
                )
        except Exception as e:
            logger.error(f"Failed to update reporter credibility: {e}")

        return credibility

