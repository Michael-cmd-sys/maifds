"""
Model explainability API endpoints for comprehensive decision transparency
"""

from flask import Flask, request, jsonify
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.logging_config import setup_logger
from src.audit.logger import get_audit_logger
from .models import (
    AuditEvent, DecisionEvent, DataAccessEvent, 
    ModelPredictionEvent, PrivacyRequestEvent,
    EventType, ComponentType, PrivacyImpact
)
from .database import AuditDatabaseManager

logger = setup_logger(__name__)


class ExplainabilityAPI:
    """API endpoints for model explainability and decision transparency"""

    def __init__(self, audit_db: Optional[AuditDatabaseManager] = None):
        """Initialize explainability API"""
        self.audit_db = audit_db or AuditDatabaseManager()
        logger.info("ExplainabilityAPI initialized")

    def explain_decision(
        self,
        decision_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None
        user_agent: Optional[str] = None
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed explanation for a specific decision"""
        try:
            # Get the decision event from audit trail
            decision_event = self.audit_db.get_audit_event_by_id(decision_id)
            
            if not decision_event:
                return {
                    "success": False,
                    "error": f"Decision {decision_id} not found"
                }
            
            # Parse decision data
            decision_data = decision_event.get("decision_data", {})
            
            # Generate comprehensive explanation
            explanation = self._generate_decision_explanation(
                decision_data=decision_data,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                component=component
            )
            
            # Log the explanation request
            self.audit_logger.log_system_event(
                event_type=EventType.DECISION_MADE,
                component=component or ComponentType.CUSTOMER_REPUTATION,
                description=f"Explanation requested for decision {decision_id}",
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                metadata={
                    "explanation_request": True,
                    "decision_id": decision_id
                }
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating decision explanation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def explain_model_prediction(
        self,
        prediction_id: str,
        model_name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        component: Optional[str] = None,
        feature_names: Optional[list] = None,
        use_shap: bool = False,
        include_counterfactual: bool = False
        include_visualization: bool = True
    ) -> Dict[str, Any]:
        """Get detailed explanation for a model prediction"""
        try:
            # Get the prediction event from audit trail
            prediction_event = self.audit_db.get_audit_event_by_id(prediction_id)
            
            if not prediction_event:
                return {
                    "success": False,
                    "error": f"Prediction {prediction_id} not found"
                }
            
            # Parse prediction data
            prediction_data = prediction_event.get("decision_data", {})
            
            # Generate comprehensive explanation
            explanation = self._generate_prediction_explanation(
                prediction_data=prediction_data,
                model_name=model_name,
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                component=component,
                feature_names=feature_names,
                use_shap=use_shap,
                include_counterfactual=include_counterfactual,
                include_visualization=include_visualization
            )
            
            # Log the explanation request
            self.audit_logger.log_model_prediction(
                model_name=model_name,
                input_features=prediction_data.get("input_features", {}),
                prediction=prediction_data.get("prediction"),
                prediction_probability=prediction_data.get("prediction_probability"),
                feature_importance=prediction_data.get("feature_importance"),
                explanation_method=prediction_data.get("explanation_method"),
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                metadata={
                    "explanation_request": True,
                    "prediction_id": prediction_id
                }
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating model prediction explanation: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_decision_explanations(
        self,
        decision_id: str,
        format: str = "json",
        days: int = 30,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get multiple decision explanations"""
        try:
            # Get decision events from audit trail
            events = self.audit_db.get_audit_events(
                event_type=EventType.DECISION_MADE,
                start_date=datetime.now() - timedelta(days=days),
                limit=limit
            )
            
            explanations = []
            for event in events:
                if event.get("decision_data"):
                    explanation = self._generate_decision_explanation(
                        decision_data=event["decision_data"],
                        user_id=event.get("user_id"),
                        session_id=event.get("session_id"),
                        ip_address=event.get("ip_address"),
                        component=event.get("component")
                    )
                    explanations.append(explanation)
            
            return {
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations),
                "format": format,
                "period_days": days,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Error getting decision explanations: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanations": []
            }

    def get_model_explanations(
        self,
        model_name: str,
        prediction_id: str,
        format: str = "json",
        days: int = 30,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get multiple model prediction explanations"""
        try:
            # Get prediction events from audit trail
            events = self.audit_db.get_audit_events(
                event_type=EventType.MODEL_PREDICTION,
                start_date=datetime.now() - timedelta(days=days),
                limit=limit
            )
            
            explanations = []
            for event in events:
                if event.get("decision_data"):
                    explanation = self._generate_prediction_explanation(
                        prediction_data=event["decision_data"],
                        model_name=model_name,
                        user_id=event.get("user_id"),
                        session_id=event.get("session_id"),
                        ip_address=event.get("ip_address"),
                        component=event.get("component"),
                        feature_names=event.get("decision_data", {}).get("feature_names"),
                        use_shap=False,
                        include_counterfactual=False,
                        include_visualization=True
                    )
                    explanations.append(explanation)
            
            return {
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations),
                "format": format,
                "period_days": days,
                "limit": limit
            }
            
        except Exception as e:
            logger.error(f"Error getting model explanations: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanations": []
            }

    def _generate_decision_explanation(
        self,
        decision_data: Dict[str, Any],
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        component: Optional[str]
    ) -> Dict[str, Any]:
        """Generate comprehensive decision explanation"""
        
        # Extract key decision factors
        risk_score = decision_data.get("risk_score", 0.0)
        confidence_score = decision_data.get("confidence_score", 0.5)
        decision_outcome = decision_data.get("decision_outcome", "unknown")
        decision_factors = decision_data.get("decision_factors", {})
        rule_triggers = decision_data.get("rule_triggers", [])
        
        # Generate explanation summary
        if decision_outcome == "block":
            explanation_summary = "Transaction blocked due to high risk factors"
            risk_level = "critical"
        elif decision_outcome == "flag":
            explanation_summary = "Transaction flagged for manual review"
            risk_level = "high"
        elif decision_outcome == "allow":
            explanation_summary = "Transaction approved - low risk detected"
            risk_level = "low"
        else:
            explanation_summary = "Transaction processed - normal risk level"
        
        # Generate detailed factor breakdown
        factor_details = []
        
        if risk_score > 0.7:
            factor_details.append(f"• High risk score ({risk_score:.3f}) indicates elevated risk")
        
        if confidence_score > 0.8:
            factor_details.append(f"• High confidence ({confidence_score:.3f}) in prediction")
        
        if rule_triggers:
            factor_details.append(f"• {len(rule_triggers)} rules were triggered")
        
        # Add decision factors to explanation
        explanation_factors = {}
        if decision_factors:
            for factor, score in decision_factors.items():
                explanation_factors[factor] = {
                    "score": score,
                    "description": self._describe_factor(factor, score)
                }
        
        # Generate visual explanation
        visual_explanation = self._generate_visual_explanation(
            decision_factors=decision_factors,
            rule_triggers=rule_triggers,
            risk_score=risk_score,
            decision_outcome=decision_outcome
        )
        
        return {
            "explanation_id": f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": "fraud_detection_system",
            "timestamp": datetime.now().isoformat(),
            "decision_data": decision_data,
            "explanation_summary": explanation_summary,
            "risk_score": risk_score,
            "confidence_score": confidence_score,
            "decision_outcome": decision_outcome,
            "decision_factors": explanation_factors,
            "visual_explanation": visual_explanation,
            "recommendations": self._generate_recommendations(risk_score, decision_outcome)
        }

    def _describe_factor(self, factor_name: str, score: float) -> str:
        """Describe a decision factor"""
        descriptions = {
            "risk_score": "Risk Score",
            "recruitment_velocity": "How quickly the agent is recruiting new merchants",
            "network_growth_rate": "Rate of network expansion",
            "transaction_anomaly": "Unusual transaction patterns detected",
            "geographic_dispersion": "Geographic spread of activities",
            "temporal_patterns": "Time-based activity patterns",
            "communication_risk": "Risk indicators in communications",
            "financial_behavior": "Financial transaction patterns",
            "association_risk": "Connections to high-risk entities"
        }
        
        return descriptions.get(factor_name, f" (score: {score:.3f}) - {descriptions.get(factor_name, 'Unknown factor')}")

    def _generate_visual_explanation(
        self,
        decision_factors: Dict[str, float],
        rule_triggers: Optional[List[str]],
        risk_score: float,
        decision_outcome: str
    ) -> Dict[str, Any]:
        """Generate visual explanation for decision"""
        
        visual_data = {
            "type": "decision_flow",
            "risk_score": risk_score,
            "decision_outcome": decision_outcome,
            "factors": decision_factors,
            "rule_triggers": rule_triggers or [],
            "recommendations": self._generate_recommendations(risk_score, decision_outcome)
        }
        
        return visual_data

    def _generate_recommendations(self, risk_score: float, decision_outcome: str) -> List[str]:
        """Generate actionable recommendations based on risk level"""
        recommendations = []
        
        if risk_score >= 0.8:
            recommendations.extend([
                "IMMEDIATE INVESTIGATION REQUIRED",
                "Consider suspending all related transactions",
                "Contact user for verification",
                "Escalate to fraud investigation team"
            ])
        elif risk_score >= 0.6:
            recommendations.extend([
                "ENHANCED MONITORING REQUIRED",
                "Increase transaction monitoring",
                "Review agent relationship patterns",
                "Consider additional verification steps"
            ])
        elif risk_score >= 0.4:
            recommendations.extend([
                "STANDARD MONITORING",
                "Continue normal monitoring procedures",
                "Review recent decision patterns"
            ])
        else:
            recommendations.extend([
                "STANDARD MONITORING",
                "Continue normal monitoring procedures"
            ])
        
        return recommendations

    def _generate_prediction_explanation(
        self,
        prediction_data: Dict[str, Any],
        model_name: str,
        user_id: Optional[str],
        session_id: Optional[str],
        ip_address: Optional[str],
        component: Optional[str],
        feature_names: Optional[List[str]] = None,
        use_shap: bool = False,
        include_counterfactual: bool = False,
        include_visualization: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive model prediction explanation"""
        
        # Extract key prediction data
        prediction = prediction_data.get("prediction", "unknown")
        prediction_probability = prediction_data.get("prediction_probability", 0.5)
        feature_importance = prediction_data.get("feature_importance", {})
        explanation_method = prediction_data.get("explanation_method", "unknown")
        
        # Generate explanation summary
        if prediction_probability >= 0.8:
            explanation_summary = f"High confidence ({prediction_probability:.3f}) - transaction blocked"
            risk_level = "critical"
        elif prediction_probability >= 0.6:
            explanation_summary = f"Medium confidence ({prediction_probability:.3f}) - transaction flagged"
            risk_level = "high"
        elif prediction_probability >= 0.4:
            explanation_summary = f"Low confidence ({prediction_probability:.3f}) - transaction monitored"
            risk_level = "medium"
        else:
            explanation_summary = f"Low confidence ({prediction_probability:.3f}) - transaction processed"
            risk_level = "low"
        
        # Generate feature importance breakdown
        feature_importance_sorted = sorted(
            feature_importance.items(), 
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Generate visual explanation
        visual_explanation = {
            "type": "feature_importance",
            "top_features": feature_importance_sorted,
            "model_name": model_name,
            "prediction": prediction,
            "confidence": prediction_probability,
            "explanation_method": explanation_method,
            "visual_explanation": {
                "type": "feature_barchart",
                "features": feature_importance_sorted,
                "values": [prediction_data.get("input_features", {}).get(name, 0) for name in feature_importance_sorted]
            }
            }
        }
        
        return {
            "explanation_id": f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "prediction_data": prediction_data,
            "explanation_summary": explanation_summary,
            "prediction_probability": prediction_probability,
            "feature_importance": feature_importance,
            "explanation_method": explanation_method,
            "visual_explanation": visual_explanation,
            "recommendations": self._generate_prediction_recommendations(prediction_probability, "high" if prediction_probability >= 0.8 else "medium" if prediction_probability >= 0.6 else "low")
        }

    def _generate_prediction_recommendations(
        self, prediction_probability: float, risk_level: str
    ) -> List[str]:
        """Generate actionable recommendations based on prediction confidence"""
        recommendations = []
        
        if prediction_probability >= 0.8:
            recommendations.extend([
                "TRANSACTION BLOCKED - High confidence prediction",
                "IMMEDIATE MANUAL REVIEW REQUIRED",
                "Contact user immediately",
                "Consider additional verification"
            ])
        elif prediction_probability >= 0.6:
            recommendations.extend([
                "TRANSACTION FLAGGED - Medium confidence prediction",
                "ENHANCED MONITORING",
                "Monitor transaction closely",
                "Consider additional verification"
            ])
        else:
            recommendations.extend([
                "TRANSACTION MONITORED - Low confidence prediction",
                "Standard monitoring procedures"
            ])
        
        return recommendations


# Create Flask app
def create_explainability_app(audit_db: Optional[AuditDatabaseManager] = None) -> Flask:
    """Create Flask app for explainability API"""
    app = Flask(__name__)
    
    # Decision explanation endpoint
    @app.route('/api/v1/explain/decision/<decision_id>', methods=['GET'])
    def explain_decision(decision_id: str):
        """Get detailed explanation for a decision"""
        try:
            api = ExplainabilityAPI(audit_db)
            return api.explain_decision(decision_id)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Model prediction explanation endpoint
    @app.route('/api/v1/explain/prediction/<prediction_id>', methods=['GET'])
    def explain_prediction(prediction_id: str):
        """Get detailed explanation for a model prediction"""
        try:
            api = ExplainabilityAPI(audit_db)
            return api.explain_model_prediction(prediction_id)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Batch explanations endpoint
    @app.route('/api/v1/explain/decisions', methods=['POST'])
    def explain_decisions(request):
        """Get explanations for multiple decisions"""
        try:
            data = request.get_json()
            decision_ids = data.get('decision_ids', [])
            
            if not decision_ids:
                return jsonify({
                    "success": False,
                    "error": "No decision IDs provided"
                }), 400
            
            api = ExplainabilityAPI(audit_db)
            explanations = []
            
            for decision_id in decision_ids:
                explanation = api.explain_decision(decision_id)
                if explanation["success"]:
                    explanations.append(explanation)
            
            return jsonify({
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations)
            }), 200
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Model predictions explanation endpoint
    @app.route('/api/v1/explain/predictions', methods=['POST'])
    def explain_predictions(request):
        """Get explanations for multiple model predictions"""
        try:
            data = request.get_json()
            prediction_ids = data.get('prediction_ids', [])
            
            if not prediction_ids:
                return jsonify({
                    "success": False,
                    "error": "No prediction IDs provided"
                }), 400
            
            api = ExplainabilityAPI(audit_db)
            explanations = []
            
            for prediction_id in prediction_ids:
                explanation = api.explain_model_prediction(prediction_id)
                if explanation["success"]:
                    explanations.append(explanation)
            
            return jsonify({
                "success": True,
                "explanations": explanations,
                "total_count": len(explanations)
            }), 200
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    # Health check endpoint
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        """Health check for explainability service"""
        try:
            api = ExplainabilityAPI(audit_db)
            
            # Check database connection
            stats = api.audit_db.get_audit_statistics(days=7)
            
            return jsonify({
                "status": "healthy",
                "service": "explainability_api",
                "database_connected": True,
                "recent_events": stats.get("total_events", 0),
                "audit_statistics": stats
            }), 200
            
        except Exception as e:
            return jsonify({
                "status": "unhealthy",
                "error": str(e)
            }), 500


def ExplainabilityAPI(audit_db: Optional[AuditDatabaseManager] = None):
    """Initialize explainability API with audit database"""
    self.audit_db = audit_db or AuditDatabaseManager()


# Global instance for easy access
_explainability_api = None


def get_explainability_api() -> ExplainabilityAPI:
    """Get global explainability API instance"""
    global _explainability_api
    if _explainability_api is None:
        _explainability_api = ExplainabilityAPI()
    return _explainability_api


# Convenience functions
def explain_decision(decision_id: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to explain a decision"""
    api = get_explainability_api()
    return api.explain_decision(decision_id, **kwargs)


def explain_model_prediction(prediction_id: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to explain a model prediction"""
    api = get_explainability_api()
    return api.explain_model_prediction(prediction_id, **kwargs)


def explain_batch_decisions(decision_ids: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function to explain multiple decisions"""
    api = get_explainability_api()
    return api.explain_decisions(decision_ids, **kwargs)


def explain_batch_predictions(prediction_ids: List[str], **kwargs) -> Dict[str, Any]:
    """Convenience function to explain multiple model predictions"""
    api = get_explainability_api()
    return api.explain_predictions(prediction_ids, **kwargs)


if __name__ == "__main__":
    # Create and run the app when script is executed directly
    app = create_explainability_app()
    app.run(host='0.0.0', port=8000, debug=False)