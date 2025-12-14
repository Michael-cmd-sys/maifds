"""
Model explainability system for ML decisions across all layers
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.logging_config import setup_logger
from src.audit.logger import get_audit_logger

logger = setup_logger(__name__)


class ModelExplainer:
    """Base class for model explainability"""

    def __init__(self, model_name: str, model_version: Optional[str] = None):
        self.model_name = model_name
        self.model_version = model_version
        self.audit_logger = get_audit_logger()

    def explain_prediction(
        self,
        input_features: Dict[str, Any],
        prediction: Any,
        prediction_probability: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for a model prediction
        
        Args:
            input_features: Input features used for prediction
            prediction: Model prediction output
            prediction_probability: Confidence score (0-1)
            feature_names: Names of features
            user_id: User identifier
            session_id: Session identifier
            entity_id: Entity being predicted
            component: System component
            
        Returns:
            Dictionary with explanation details
        """
        raise NotImplementedError("Subclasses must implement explain_prediction")

    def generate_counterfactual(
        self,
        input_features: Dict[str, Any],
        original_prediction: Any,
        counterfactual_features: Dict[str, Any],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate counterfactual analysis
        
        Args:
            input_features: Original input features
            original_prediction: Original model prediction
            counterfactual_features: Modified features for counterfactual
            feature_names: Names of features
            
        Returns:
            Dictionary with counterfactual analysis
        """
        raise NotImplementedError("Subclasses must implement generate_counterfactual")


class RuleBasedExplainer(ModelExplainer):
    """Explainer for rule-based systems"""

    def explain_prediction(
        self,
        input_features: Dict[str, Any],
        prediction: str,
        rule_triggers: Optional[List[str]] = None,
        rule_weights: Optional[Dict[str, float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Explain rule-based decision"""
        
        explanation = {
            "explanation_id": f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation_method": "rule_based",
            "timestamp": datetime.now().isoformat(),
            "input_features": input_features,
            "prediction": prediction,
            "rule_triggers": rule_triggers or [],
            "rule_weights": rule_weights or {},
            "explanation_summary": self._generate_rule_summary(rule_triggers, input_features),
            "feature_importance": self._calculate_rule_importance(rule_triggers, rule_weights),
            "confidence_score": 1.0 if rule_triggers else 0.5,  # High confidence if rules triggered
            "counterfactual_analysis": None,
            "visual_explanation": self._generate_rule_visualization(rule_triggers, input_features)
        }
        
        # Log the explanation
        self.audit_logger.log_model_prediction(
            model_name=self.model_name,
            input_features=input_features,
            prediction=prediction,
            prediction_probability=explanation["confidence_score"],
            feature_importance=explanation["feature_importance"],
            explanation_method=explanation["explanation_method"],
            user_id=user_id,
            session_id=session_id,
            entity_id=entity_id,
            component=component
        )
        
        return explanation

    def _generate_rule_summary(
        self, rule_triggers: Optional[List[str]], input_features: Dict[str, Any]
    ) -> str:
        """Generate human-readable rule summary"""
        if not rule_triggers:
            return "No rules triggered - decision based on default behavior"
        
        triggered_rules = []
        for rule in rule_triggers:
            triggered_rules.append(f"• {rule}")
        
        return f"Decision based on {len(rule_triggers)} triggered rules: {' '.join(triggered_rules)}"

    def _calculate_rule_importance(
        self, rule_triggers: Optional[List[str]], rule_weights: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate feature importance based on triggered rules"""
        if not rule_triggers or not rule_weights:
            return {}
        
        importance = {}
        for rule in rule_triggers:
            weight = rule_weights.get(rule, 1.0)
            # Map rule to relevant features (simplified)
            if "high_amount" in rule.lower():
                importance["amount"] = importance.get("amount", 0) + weight
            if "new_recipient" in rule.lower():
                importance["recipient_new"] = importance.get("recipient_new", 0) + weight
            if "suspicious_timing" in rule.lower():
                importance["timing"] = importance.get("timing", 0) + weight
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v/total_importance for k, v in importance.items()}
        
        return importance

    def _generate_rule_visualization(
        self, rule_triggers: Optional[List[str]], input_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visualization data for rule-based explanation"""
        if not rule_triggers:
            return {"type": "no_rules", "message": "No rules were triggered"}
        
        return {
            "type": "rule_flow",
            "triggered_rules": rule_triggers,
            "rule_flow": self._build_rule_flow(rule_triggers),
            "feature_values": {
                "amount": input_features.get("amount", 0),
                "recipient_new": input_features.get("recipient_first_time", 0),
                "timing": input_features.get("call_to_tx_delta_seconds", 0)
            }
        }

    def _build_rule_flow(self, rule_triggers: List[str]) -> List[Dict[str, Any]]:
        """Build rule flow visualization"""
        flow = []
        
        rule_descriptions = {
            "high_amount": "High transaction amount detected",
            "new_recipient": "First-time recipient detected",
            "suspicious_timing": "Suspicious timing detected"
        }
        
        for i, rule in enumerate(rule_triggers):
            flow.append({
                "step": i + 1,
                "rule": rule,
                "description": rule_descriptions.get(rule, "Unknown rule"),
                "status": "triggered"
            })
        
        return flow


class MindSporeExplainer(ModelExplainer):
    """Explainer for MindSpore neural network models"""

    def __init__(self, model_name: str, model=None):
        super().__init__(model_name)
        self.model = model
        self.feature_names = None

    def explain_prediction(
        self,
        input_features: Dict[str, Any],
        prediction: Any,
        prediction_probability: Optional[float] = None,
        feature_names: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        component: Optional[str] = None,
        use_shap: bool = True
    ) -> Dict[str, Any]:
        """Explain MindSpore model prediction using SHAP-like approach"""
        
        if self.model is None:
            return self._fallback_explanation(input_features, prediction, user_id, session_id, entity_id, component)
        
        try:
            # Convert input features to numpy array
            if feature_names is None:
                feature_names = list(input_features.keys())
            
            feature_values = []
            for name in feature_names:
                value = input_features.get(name, 0)
                feature_values.append(float(value))
            
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Generate explanation using gradient-based approach (simplified SHAP)
            explanation = self._generate_gradient_explanation(
                feature_array, feature_names, prediction, prediction_probability
            )
            
            # Add counterfactual analysis
            counterfactual = self._generate_counterfactual_analysis(
                feature_array, feature_names, prediction
            )
            explanation["counterfactual_analysis"] = counterfactual
            
            # Log the explanation
            self.audit_logger.log_model_prediction(
                model_name=self.model_name,
                input_features=input_features,
                prediction=prediction,
                prediction_probability=prediction_probability,
                feature_importance=explanation["feature_importance"],
                explanation_method=explanation["explanation_method"],
                user_id=user_id,
                session_id=session_id,
                entity_id=entity_id,
                component=component
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating MindSpore explanation: {e}")
            return self._fallback_explanation(input_features, prediction, user_id, session_id, entity_id, component)

    def _generate_gradient_explanation(
        self, feature_array: np.ndarray, feature_names: List[str], 
        prediction: Any, prediction_probability: Optional[float]
    ) -> Dict[str, Any]:
        """Generate gradient-based feature importance"""
        
        # Simplified gradient-based importance (placeholder for actual SHAP)
        # In a real implementation, this would use actual model gradients
        
        # Generate mock feature importance based on feature values
        importance_scores = []
        for i in range(len(feature_names)):
            # Higher absolute values get higher importance (simplified heuristic)
            feature_value = abs(feature_array[0, i])
            importance_scores.append(feature_value)
        
        # Normalize importance scores
        total_importance = sum(importance_scores)
        if total_importance > 0:
            importance_scores = [score/total_importance for score in importance_scores]
        else:
            importance_scores = [1.0/len(feature_names)] * len(feature_names)
        
        feature_importance = dict(zip(feature_names, importance_scores))
        
        # Generate explanation summary
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        explanation_summary = f"Top contributing factors: {', '.join([f'{name} ({score:.3f})' for name, score in top_features])}"
        
        return {
            "explanation_id": f"gradient_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation_method": "gradient_based",
            "timestamp": datetime.now().isoformat(),
            "input_features": dict(zip(feature_names, feature_array[0].tolist())),
            "prediction": prediction,
            "prediction_probability": prediction_probability,
            "feature_importance": feature_importance,
            "explanation_summary": explanation_summary,
            "top_features": top_features,
            "visual_explanation": {
                "type": "feature_importance",
                "feature_names": feature_names,
                "importance_scores": importance_scores,
                "top_features": top_features
            }
        }

    def _generate_counterfactual_analysis(
        self, feature_array: np.ndarray, feature_names: List[str], prediction: Any
    ) -> Dict[str, Any]:
        """Generate counterfactual analysis"""
        
        counterfactuals = []
        
        # Generate counterfactuals by modifying top features
        importance_scores = []
        for i in range(len(feature_names)):
            feature_value = abs(feature_array[0, i])
            importance_scores.append(feature_value)
        
        # Get top 3 most important features
        top_indices = sorted(range(len(importance_scores)), key=lambda x: importance_scores[x], reverse=True)[:3]
        
        for idx in top_indices:
            cf_feature_array = feature_array.copy()
            cf_feature_array[0, idx] = 0  # Set to zero
            
            # Try to get counterfactual prediction (simplified)
            cf_prediction = "not_fraud" if prediction == "fraud" else "fraud"
            
            counterfactuals.append({
                "feature_name": feature_names[idx],
                "original_value": float(feature_array[0, idx]),
                "counterfactual_value": 0.0,
                "counterfactual_prediction": cf_prediction,
                "change_impact": "high" if importance_scores[idx] > 0.5 else "medium"
            })
        
        return {
            "counterfactuals": counterfactuals,
            "method": "feature_perturbation",
            "summary": f"Generated {len(counterfactuals)} counterfactual examples by modifying top features"
        }

    def _fallback_explanation(
        self, input_features: Dict[str, Any], prediction: Any,
        user_id: Optional[str], session_id: Optional[str], 
        entity_id: Optional[str], component: Optional[str]
    ) -> Dict[str, Any]:
        """Fallback explanation when model is not available"""
        
        return {
            "explanation_id": f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation_method": "rule_based_fallback",
            "timestamp": datetime.now().isoformat(),
            "input_features": input_features,
            "prediction": prediction,
            "prediction_probability": 0.5,  # Default confidence
            "feature_importance": {},
            "explanation_summary": "Model not available - using rule-based fallback",
            "visual_explanation": {
                "type": "model_unavailable",
                "message": "Model not loaded for detailed explanation"
            }
        }


class EnsembleExplainer(ModelExplainer):
    """Explainer for ensemble models (rules + ML)"""

    def __init__(self, model_name: str, rule_explainer: RuleBasedExplainer, 
                 ml_explainer: MindSporeExplainer):
        super().__init__(model_name)
        self.rule_explainer = rule_explainer
        self.ml_explainer = ml_explainer

    def explain_prediction(
        self,
        input_features: Dict[str, Any],
        prediction: str,
        prediction_probability: Optional[float] = None,
        rule_triggers: Optional[List[str]] = None,
        rule_weights: Optional[Dict[str, float]] = None,
        ml_features: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Any] = None,
        ml_probability: Optional[float] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        component: Optional[str] = None
    ) -> Dict[str, Any]:
        """Explain ensemble prediction combining rules and ML"""
        
        explanations = {}
        
        # Get rule-based explanation
        if rule_triggers:
            explanations["rule_based"] = self.rule_explainer.explain_prediction(
                input_features=input_features,
                prediction=prediction,
                rule_triggers=rule_triggers,
                rule_weights=rule_weights,
                user_id=user_id,
                session_id=session_id,
                entity_id=entity_id,
                component=component
            )
        
        # Get ML explanation
        if ml_features and ml_prediction is not None:
            explanations["ml_based"] = self.ml_explainer.explain_prediction(
                input_features=ml_features,
                prediction=ml_prediction,
                prediction_probability=ml_probability,
                user_id=user_id,
                session_id=session_id,
                entity_id=entity_id,
                component=component
            )
        
        # Combine explanations
        combined_importance = {}
        combined_summary = []
        
        if "rule_based" in explanations:
            combined_importance.update(explanations["rule_based"]["feature_importance"])
            combined_summary.append(explanations["rule_based"]["explanation_summary"])
        
        if "ml_based" in explanations:
            combined_importance.update(explanations["ml_based"]["feature_importance"])
            combined_summary.append(explanations["ml_based"]["explanation_summary"])
        
        # Calculate combined confidence
        rule_confidence = explanations.get("rule_based", {}).get("confidence_score", 0.5)
        ml_confidence = explanations.get("ml_based", {}).get("confidence_score", 0.5)
        combined_confidence = (rule_confidence + ml_confidence) / 2
        
        final_summary = " | ".join(combined_summary) if combined_summary else "No explanation available"
        
        explanation = {
            "explanation_id": f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_name": self.model_name,
            "model_version": self.model_version,
            "explanation_method": "ensemble",
            "timestamp": datetime.now().isoformat(),
            "input_features": input_features,
            "prediction": prediction,
            "prediction_probability": prediction_probability or combined_confidence,
            "feature_importance": combined_importance,
            "explanation_summary": final_summary,
            "explanations": explanations,
            "visual_explanation": {
                "type": "ensemble",
                "rule_explanation": explanations.get("rule_based"),
                "ml_explanation": explanations.get("ml_based"),
                "combined_importance": combined_importance
            }
        }
        
        # Log the ensemble explanation
        self.audit_logger.log_model_prediction(
            model_name=self.model_name,
            input_features=input_features,
            prediction=prediction,
            prediction_probability=explanation["prediction_probability"],
            feature_importance=explanation["feature_importance"],
            explanation_method=explanation["explanation_method"],
            user_id=user_id,
            session_id=session_id,
            entity_id=entity_id,
            component=component
        )
        
        return explanation


# Factory function for creating explainers
def create_explainer(model_type: str, model_name: str, **kwargs) -> ModelExplainer:
    """Factory function to create appropriate explainer"""
    
    if model_type.lower() == "rule_based":
        return RuleBasedExplainer(model_name, **kwargs)
    elif model_type.lower() == "mindspore":
        return MindSporeExplainer(model_name, **kwargs)
    elif model_type.lower() == "ensemble":
        rule_explainer = kwargs.get("rule_explainer")
        ml_explainer = kwargs.get("ml_explainer")
        return EnsembleExplainer(model_name, rule_explainer, ml_explainer)
    else:
        # Default to rule-based
        return RuleBasedExplainer(model_name, **kwargs)