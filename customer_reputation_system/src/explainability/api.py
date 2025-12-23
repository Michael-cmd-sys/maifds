"""
Model explainability API endpoints for comprehensive decision transparency.

NOTE:
- This module provides a small Flask app (optional) and a pure-Python API class.
- It depends on the audit database manager under customer_reputation_system.src.audit.
- It is written to compile cleanly even if some optional fields are missing in stored events.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request

from customer_reputation_system.config.logging_config import setup_logger
from customer_reputation_system.src.audit.logger import get_audit_logger
from customer_reputation_system.src.audit.models import EventType, ComponentType
from customer_reputation_system.src.audit.database import AuditDatabaseManager

logger = setup_logger(__name__)


class ExplainabilityAPI:
    """API for model explainability and decision transparency."""

    def __init__(self, audit_db: Optional[AuditDatabaseManager] = None):
        self.audit_db = audit_db or AuditDatabaseManager()
        self.audit_logger = get_audit_logger()
        logger.info("ExplainabilityAPI initialized")

    # -------------------------------------------------------------------------
    # Public APIs
    # -------------------------------------------------------------------------
    def explain_decision(
        self,
        decision_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        component: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get detailed explanation for a specific decision audit event."""
        try:
            event = self.audit_db.get_audit_event_by_id(decision_id)
            if not event:
                return {"success": False, "error": f"Decision {decision_id} not found"}

            decision_data = event.get("decision_data") or event.get("metadata") or {}
            explanation = self._generate_decision_explanation(
                decision_id=decision_id,
                decision_data=decision_data,
                user_id=user_id or event.get("user_id"),
                session_id=session_id or event.get("session_id"),
                ip_address=ip_address or event.get("ip_address"),
                component=component or event.get("component"),
            )

            self._log_explanation_request(
                kind="decision",
                entity_id=decision_id,
                user_id=user_id or event.get("user_id"),
                session_id=session_id or event.get("session_id"),
                ip_address=ip_address or event.get("ip_address"),
                user_agent=user_agent,
                component=component or event.get("component"),
            )

            return explanation
        except Exception as e:
            logger.exception("Error generating decision explanation")
            return {"success": False, "error": str(e)}

    def explain_model_prediction(
        self,
        prediction_id: str,
        model_name: str = "unknown_model",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        component: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        use_shap: bool = False,
        include_counterfactual: bool = False,
        include_visualization: bool = True,
    ) -> Dict[str, Any]:
        """Get detailed explanation for a model prediction audit event."""
        try:
            event = self.audit_db.get_audit_event_by_id(prediction_id)
            if not event:
                return {"success": False, "error": f"Prediction {prediction_id} not found"}

            prediction_data = event.get("decision_data") or event.get("metadata") or {}
            explanation = self._generate_prediction_explanation(
                prediction_id=prediction_id,
                prediction_data=prediction_data,
                model_name=model_name,
                user_id=user_id or event.get("user_id"),
                session_id=session_id or event.get("session_id"),
                ip_address=ip_address or event.get("ip_address"),
                component=component or event.get("component"),
                feature_names=feature_names,
                use_shap=use_shap,
                include_counterfactual=include_counterfactual,
                include_visualization=include_visualization,
            )

            self._log_explanation_request(
                kind="prediction",
                entity_id=prediction_id,
                user_id=user_id or event.get("user_id"),
                session_id=session_id or event.get("session_id"),
                ip_address=ip_address or event.get("ip_address"),
                user_agent=user_agent,
                component=component or event.get("component"),
                extra={"model_name": model_name},
            )

            return explanation
        except Exception as e:
            logger.exception("Error generating model prediction explanation")
            return {"success": False, "error": str(e)}

    def get_decision_explanations(self, days: int = 30, limit: int = 100) -> Dict[str, Any]:
        """Batch: explanations for decision events within a time range."""
        try:
            events = self.audit_db.get_audit_events(
                event_type=EventType.DECISION_MADE,
                start_date=datetime.now() - timedelta(days=days),
                limit=limit,
            )
            explanations = []
            for ev in events:
                decision_id = ev.get("event_id") or ev.get("id") or "unknown"
                decision_data = ev.get("decision_data") or ev.get("metadata") or {}
                explanations.append(
                    self._generate_decision_explanation(
                        decision_id=str(decision_id),
                        decision_data=decision_data,
                        user_id=ev.get("user_id"),
                        session_id=ev.get("session_id"),
                        ip_address=ev.get("ip_address"),
                        component=ev.get("component"),
                    )
                )
            return {
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations),
                "period_days": days,
                "limit": limit,
            }
        except Exception as e:
            logger.exception("Error getting decision explanations")
            return {"success": False, "error": str(e), "explanations": []}

    def get_model_explanations(
        self,
        model_name: str,
        days: int = 30,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Batch: explanations for model prediction events within a time range."""
        try:
            events = self.audit_db.get_audit_events(
                event_type=EventType.MODEL_PREDICTION,
                start_date=datetime.now() - timedelta(days=days),
                limit=limit,
            )
            explanations = []
            for ev in events:
                prediction_id = ev.get("event_id") or ev.get("id") or "unknown"
                prediction_data = ev.get("decision_data") or ev.get("metadata") or {}
                explanations.append(
                    self._generate_prediction_explanation(
                        prediction_id=str(prediction_id),
                        prediction_data=prediction_data,
                        model_name=model_name,
                        user_id=ev.get("user_id"),
                        session_id=ev.get("session_id"),
                        ip_address=ev.get("ip_address"),
                        component=ev.get("component"),
                        feature_names=prediction_data.get("feature_names"),
                        use_shap=False,
                        include_counterfactual=False,
                        include_visualization=True,
                    )
                )

            return {
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations),
                "period_days": days,
                "limit": limit,
                "model_name": model_name,
            }
        except Exception as e:
            logger.exception("Error getting model explanations")
            return {"success": False, "error": str(e), "explanations": []}

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------
    def _log_explanation_request(
        self,
        kind: str,
        entity_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        user_agent: Optional[str],
        component: Optional[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log that an explanation was requested (non-fatal)."""
        try:
            self.audit_logger.log_system_event(
                event_type=EventType.DECISION_MADE,
                component=(component or ComponentType.CUSTOMER_REPUTATION),
                description=f"Explanation requested ({kind}) for {entity_id}",
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                metadata={
                    "explanation_request": True,
                    "kind": kind,
                    "entity_id": entity_id,
                    "user_agent": user_agent,
                    **(extra or {}),
                },
            )
        except Exception as e:
            logger.debug(f"Explainability request log skipped: {e}")

    def _generate_decision_explanation(
        self,
        decision_id: str,
        decision_data: Dict[str, Any],
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        component: Optional[str],
    ) -> Dict[str, Any]:
        risk_score = float(decision_data.get("risk_score", 0.0) or 0.0)
        confidence_score = float(decision_data.get("confidence_score", 0.5) or 0.5)
        decision_outcome = str(decision_data.get("decision_outcome", "unknown") or "unknown")
        decision_factors = decision_data.get("decision_factors") or {}
        rule_triggers = decision_data.get("rule_triggers") or []

        summary, risk_level = self._decision_summary(decision_outcome, risk_score)

        explanation_factors: Dict[str, Any] = {}
        if isinstance(decision_factors, dict):
            for factor, score in decision_factors.items():
                try:
                    s = float(score)
                except Exception:
                    s = 0.0
                explanation_factors[str(factor)] = {
                    "score": s,
                    "description": self._describe_factor(str(factor), s),
                }

        visual_explanation = self._generate_visual_explanation(
            decision_factors=decision_factors if isinstance(decision_factors, dict) else {},
            rule_triggers=rule_triggers if isinstance(rule_triggers, list) else [],
            risk_score=risk_score,
            decision_outcome=decision_outcome,
        )

        return {
            "success": True,
            "explanation_id": f"decision_{decision_id}",
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "risk_level": risk_level,
            "explanation_summary": summary,
            "risk_score": risk_score,
            "confidence_score": confidence_score,
            "decision_outcome": decision_outcome,
            "decision_factors": explanation_factors,
            "rule_triggers": rule_triggers,
            "visual_explanation": visual_explanation,
            "recommendations": self._generate_recommendations(risk_score, decision_outcome),
            "raw": decision_data,
        }

    def _generate_prediction_explanation(
        self,
        prediction_id: str,
        prediction_data: Dict[str, Any],
        model_name: str,
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        component: Optional[str],
        feature_names: Optional[List[str]] = None,
        use_shap: bool = False,
        include_counterfactual: bool = False,
        include_visualization: bool = True,
    ) -> Dict[str, Any]:
        prediction = prediction_data.get("prediction", "unknown")
        prob = float(prediction_data.get("prediction_probability", 0.5) or 0.5)
        feature_importance = prediction_data.get("feature_importance") or {}
        method = prediction_data.get("explanation_method") or ("shap" if use_shap else "heuristic")

        summary, risk_level = self._prediction_summary(prob)

        top_features: List[Tuple[str, float]] = []
        if isinstance(feature_importance, dict):
            items = []
            for k, v in feature_importance.items():
                try:
                    items.append((str(k), float(v)))
                except Exception:
                    items.append((str(k), 0.0))
            top_features = sorted(items, key=lambda x: x[1], reverse=True)[:5]

        viz = None
        if include_visualization:
            input_features = prediction_data.get("input_features") or {}
            viz = {
                "type": "feature_importance",
                "top_features": top_features,
                "feature_values": [
                    (name, input_features.get(name)) for name, _ in top_features
                ],
            }

        # (Optional) counterfactual stub – kept simple to avoid heavy dependencies.
        counterfactual = None
        if include_counterfactual:
            counterfactual = {
                "available": False,
                "reason": "Counterfactual generation not enabled (no library configured)",
            }

        return {
            "success": True,
            "explanation_id": f"prediction_{prediction_id}",
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "model_name": model_name,
            "risk_level": risk_level,
            "explanation_summary": summary,
            "prediction": prediction,
            "prediction_probability": prob,
            "explanation_method": method,
            "top_features": top_features,
            "visual_explanation": viz,
            "counterfactual": counterfactual,
            "recommendations": self._generate_prediction_recommendations(prob),
            "raw": prediction_data,
        }

    def _decision_summary(self, decision_outcome: str, risk_score: float) -> Tuple[str, str]:
        outcome = decision_outcome.lower()
        if outcome == "block":
            return "Transaction blocked due to high risk factors", "critical"
        if outcome == "flag":
            return "Transaction flagged for manual review", "high"
        if outcome == "allow":
            return "Transaction approved - low risk detected", "low"
        # fallback
        if risk_score >= 0.8:
            return "High risk detected", "critical"
        if risk_score >= 0.6:
            return "Elevated risk detected", "high"
        if risk_score >= 0.4:
            return "Moderate risk detected", "medium"
        return "Low risk detected", "low"

    def _prediction_summary(self, prob: float) -> Tuple[str, str]:
        if prob >= 0.8:
            return f"High confidence ({prob:.3f}) prediction", "critical"
        if prob >= 0.6:
            return f"Medium confidence ({prob:.3f}) prediction", "high"
        if prob >= 0.4:
            return f"Low confidence ({prob:.3f}) prediction", "medium"
        return f"Very low confidence ({prob:.3f}) prediction", "low"

    def _describe_factor(self, factor_name: str, score: float) -> str:
        descriptions = {
            "recruitment_velocity": "How quickly the agent is recruiting new merchants",
            "network_growth_rate": "Rate of network expansion",
            "transaction_anomaly_score": "Unusual transaction patterns detected",
            "geographic_dispersion": "Geographic spread of activities",
            "temporal_patterns": "Time-based activity patterns",
            "communication_risk": "Risk indicators in communications",
            "financial_behavior_score": "Financial transaction patterns",
            "association_risk": "Connections to high-risk entities",
        }
        base = descriptions.get(factor_name, "Unknown factor")
        return f"{base} (score: {score:.3f})"

    def _generate_visual_explanation(
        self,
        decision_factors: Dict[str, float],
        rule_triggers: List[str],
        risk_score: float,
        decision_outcome: str,
    ) -> Dict[str, Any]:
        return {
            "type": "decision_flow",
            "risk_score": risk_score,
            "decision_outcome": decision_outcome,
            "factors": decision_factors,
            "rule_triggers": rule_triggers,
        }

    def _generate_recommendations(self, risk_score: float, decision_outcome: str) -> List[str]:
        recs: List[str] = []
        if risk_score >= 0.8:
            recs += [
                "IMMEDIATE INVESTIGATION REQUIRED",
                "Consider suspending related transactions",
                "Contact user for verification",
                "Escalate to fraud investigation team",
            ]
        elif risk_score >= 0.6:
            recs += [
                "ENHANCED MONITORING REQUIRED",
                "Increase monitoring frequency",
                "Review relationship patterns",
                "Consider additional verification steps",
            ]
        elif risk_score >= 0.4:
            recs += [
                "STANDARD MONITORING",
                "Review recent decision patterns",
            ]
        else:
            recs += ["STANDARD MONITORING"]
        return recs

    def _generate_prediction_recommendations(self, prediction_probability: float) -> List[str]:
        if prediction_probability >= 0.8:
            return [
                "TRANSACTION BLOCKED - High confidence prediction",
                "IMMEDIATE MANUAL REVIEW REQUIRED",
                "Contact user immediately",
            ]
        if prediction_probability >= 0.6:
            return [
                "TRANSACTION FLAGGED - Medium confidence prediction",
                "ENHANCED MONITORING",
                "Consider additional verification",
            ]
        return [
            "TRANSACTION MONITORED - Low confidence prediction",
            "Standard monitoring procedures",
        ]


# -----------------------------------------------------------------------------
# Flask app factory (optional)
# -----------------------------------------------------------------------------
def create_explainability_app(audit_db: Optional[AuditDatabaseManager] = None) -> Flask:
    app = Flask(__name__)
    api = ExplainabilityAPI(audit_db)

    @app.get("/api/v1/explain/decision/<decision_id>")
    def explain_decision_endpoint(decision_id: str):
        result = api.explain_decision(
            decision_id=decision_id,
            user_id=request.args.get("user_id"),
            session_id=request.args.get("session_id"),
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
            component=request.args.get("component"),
        )
        return jsonify(result), (200 if result.get("success") else 404)

    @app.get("/api/v1/explain/prediction/<prediction_id>")
    def explain_prediction_endpoint(prediction_id: str):
        result = api.explain_model_prediction(
            prediction_id=prediction_id,
            model_name=request.args.get("model_name", "unknown_model"),
            user_id=request.args.get("user_id"),
            session_id=request.args.get("session_id"),
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
            component=request.args.get("component"),
        )
        return jsonify(result), (200 if result.get("success") else 404)

    @app.post("/api/v1/explain/decisions")
    def explain_decisions_batch():
        data = request.get_json(silent=True) or {}
        ids = data.get("decision_ids") or []
        if not isinstance(ids, list) or not ids:
            return jsonify({"success": False, "error": "No decision IDs provided"}), 400

        explanations = []
        for did in ids:
            res = api.explain_decision(str(did))
            if res.get("success"):
                explanations.append(res)

        return jsonify({"success": True, "explanations": explanations, "total_count": len(explanations)}), 200

    @app.post("/api/v1/explain/predictions")
    def explain_predictions_batch():
        data = request.get_json(silent=True) or {}
        ids = data.get("prediction_ids") or []
        model_name = data.get("model_name", "unknown_model")

        if not isinstance(ids, list) or not ids:
            return jsonify({"success": False, "error": "No prediction IDs provided"}), 400

        explanations = []
        for pid in ids:
            res = api.explain_model_prediction(str(pid), model_name=model_name)
            if res.get("success"):
                explanations.append(res)

        return jsonify({"success": True, "explanations": explanations, "total_count": len(explanations)}), 200

    @app.get("/api/v1/health")
    def health_check():
        try:
            stats = api.audit_db.get_audit_statistics(days=7)
            return jsonify(
                {
                    "status": "healthy",
                    "service": "explainability_api",
                    "database_connected": True,
                    "recent_events": stats.get("total_events", 0),
                    "audit_statistics": stats,
                }
            ), 200
        except Exception as e:
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    return app


# Global singleton
_explainability_api: Optional[ExplainabilityAPI] = None


def get_explainability_api() -> ExplainabilityAPI:
    global _explainability_api
    if _explainability_api is None:
        _explainability_api = ExplainabilityAPI()
    return _explainability_api


if __name__ == "__main__":
    app = create_explainability_app()
    app.run(host="0.0.0.0", port=8000, debug=False)
