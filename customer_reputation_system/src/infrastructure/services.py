"""Service layer interfaces for the customer reputation system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from uuid import UUID

from ..infrastructure.result import Result
from ..ingestion.report_handler import ReportHandler
from ..credibility.calculator import CredibilityCalculator
from ..reputation.calculator import ReputationCalculator
from ..nlp.text_analyzer import TextAnalyzer


class ReportService(ABC):
    """Interface for report processing services."""
    
    @abstractmethod
    async def submit_report(self, report_data: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """Submit a new report for processing."""
        pass
    
    @abstractmethod
    async def get_report(self, report_id: UUID) -> Result[Optional[Dict[str, Any]]]:
        """Get a report by ID."""
        pass
    
    @abstractmethod
    async def update_report(self, report_id: UUID, updates: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """Update an existing report."""
        pass
    
    @abstractmethod
    async def delete_report(self, report_id: UUID) -> Result[bool]:
        """Delete a report."""
        pass


class CredibilityService(ABC):
    """Interface for reporter credibility services."""
    
    @abstractmethod
    async def calculate_credibility(self, reporter_id: str) -> Result[float]:
        """Calculate credibility score for a reporter."""
        pass
    
    @abstractmethod
    async def update_credibility(self, reporter_id: str, report_data: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any]]:
        """Update credibility score for a reporter."""
        pass
    
    @abstractmethod
    async def get_credibility_history(self, reporter_id: str) -> Result[List[Dict[str, Any]]]:
        """Get credibility score history for a reporter."""
        pass


class ReputationService(ABC):
    """Interface for merchant reputation services."""
    
    @abstractmethod
    async def calculate_reputation(self, merchant_id: str) -> Result[float]:
        """Calculate reputation score for a merchant."""
        pass
    
    @abstractmethod
    async def update_reputation(self, merchant_id: str, report_data: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any]]:
        """Update reputation score for a merchant."""
        pass
    
    @abstractmethod
    async def get_reputation_history(self, merchant_id: str) -> Result[List[Dict[str, Any]]]:
        """Get reputation score history for a merchant."""
        pass


class NLPService(ABC):
    """Interface for NLP analysis services."""
    
    @abstractmethod
    async def analyze_text(self, text: str) -> Result[Dict[str, Any]]:
        """Analyze text for sentiment, urgency, and credibility."""
        pass
    
    @abstractmethod
    async def batch_analyze(self, texts: List[str]) -> Result[List[Dict[str, Any]]]:
        """Analyze multiple texts in batch."""
        pass


class NotificationService(ABC):
    """Interface for notification services."""
    
    @abstractmethod
    async def send_alert(self, alert_data: Dict[str, Any]) -> Result[bool]:
        """Send an alert notification."""
        pass
    
    @abstractmethod
    async def send_report_confirmation(self, reporter_id: str, report_id: UUID) -> Result[bool]:
        """Send report confirmation to reporter."""
        pass


class ReportProcessingService(ABC):
    """Interface for comprehensive report processing."""
    
    @abstractmethod
    async def process_report(self, report_data: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """Process a complete report with all analysis."""
        pass
    
    @abstractmethod
    async def batch_process_reports(self, reports: List[Dict[str, Any]]) -> Result[List[Dict[str, Any]]]:
        """Process multiple reports in batch."""
        pass
    
    @abstractmethod
    async def validate_report(self, report_data: Dict[str, Any]) -> Result[bool]:
        """Validate report data before processing."""
        pass


class ReportHandlerService(ReportProcessingService):
    """Implementation of report processing service using the existing ReportHandler."""
    
    def __init__(self, report_handler: ReportHandler):
        self.report_handler = report_handler
    
    async def process_report(self, report_data: Dict[str, Any]) -> Result[Dict[str, Any]]:
        """Process a complete report with all analysis."""
        try:
            result = self.report_handler.submit_report(report_data)
            return Result.success(result)
        except Exception as e:
            return Result.failure(e)
    
    async def batch_process_reports(self, reports: List[Dict[str, Any]]) -> Result[List[Dict[str, Any]]]:
        """Process multiple reports in batch."""
        results = []
        errors = []
        
        for report in reports:
            try:
                result = self.report_handler.submit_report(report)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        if errors:
            return Result.failure(Exception(f"Batch processing failed: {errors}"))
        
        return Result.success(results)
    
    async def validate_report(self, report_data: Dict[str, Any]) -> Result[bool]:
        """Validate report data before processing."""
        try:
            # Basic validation - could be expanded
            required_fields = ["reporter_id", "merchant_id", "report_type"]
            for field in required_fields:
                if field not in report_data:
                    return Result.failure(ValueError(f"Missing required field: {field}"))
            
            return Result.success(True)
        except Exception as e:
            return Result.failure(e)


class CredibilityCalculationService(CredibilityService):
    """Implementation of credibility service using CredibilityCalculator."""
    
    def __init__(self, credibility_calculator: CredibilityCalculator):
        self.calculator = credibility_calculator
    
    async def calculate_credibility(self, reporter_id: str) -> Result[float]:
        """Calculate credibility score for a reporter."""
        try:
            credibility = self.calculator.calculate_credibility(reporter_id)
            return Result.success(credibility.credibility_score)
        except Exception as e:
            return Result.failure(e)
    
    async def update_credibility(self, reporter_id: str, report_data: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any]]:
        """Update credibility score for a reporter."""
        try:
            credibility = self.calculator.calculate_credibility(reporter_id, report_data)
            return Result.success(credibility.to_dict())
        except Exception as e:
            return Result.failure(e)
    
    async def get_credibility_history(self, reporter_id: str) -> Result[List[Dict[str, Any]]]:
        """Get credibility score history for a reporter."""
        # This would need to be implemented in the calculator
        # For now, return current credibility
        try:
            credibility = self.calculator.calculate_credibility(reporter_id)
            return Result.success([credibility.to_dict()])
        except Exception as e:
            return Result.failure(e)


class ReputationCalculationService(ReputationService):
    """Implementation of reputation service using ReputationCalculator."""
    
    def __init__(self, reputation_calculator: ReputationCalculator):
        self.calculator = reputation_calculator
    
    async def calculate_reputation(self, merchant_id: str) -> Result[float]:
        """Calculate reputation score for a merchant."""
        try:
            reputation = self.calculator.calculate_reputation(merchant_id)
            return Result.success(reputation.reputation_score)
        except Exception as e:
            return Result.failure(e)
    
    async def update_reputation(self, merchant_id: str, report_data: Optional[Dict[str, Any]] = None) -> Result[Dict[str, Any]]:
        """Update reputation score for a merchant."""
        try:
            reputation = self.calculator.calculate_reputation(merchant_id, report_data)
            return Result.success(reputation.to_dict())
        except Exception as e:
            return Result.failure(e)
    
    async def get_reputation_history(self, merchant_id: str) -> Result[List[Dict[str, Any]]]:
        """Get reputation score history for a merchant."""
        # This would need to be implemented in the calculator
        # For now, return current reputation
        try:
            reputation = self.calculator.calculate_reputation(merchant_id)
            return Result.success([reputation.to_dict()])
        except Exception as e:
            return Result.failure(e)


class TextAnalysisService(NLPService):
    """Implementation of NLP service using TextAnalyzer."""
    
    def __init__(self, text_analyzer: TextAnalyzer):
        self.analyzer = text_analyzer
    
    async def analyze_text(self, text: str) -> Result[Dict[str, Any]]:
        """Analyze text for sentiment, urgency, and credibility."""
        try:
            analysis = self.analyzer.analyze(text)
            return Result.success(analysis)
        except Exception as e:
            return Result.failure(e)
    
    async def batch_analyze(self, texts: List[str]) -> Result[List[Dict[str, Any]]]:
        """Analyze multiple texts in batch."""
        results = []
        errors = []
        
        for text in texts:
            try:
                analysis = self.analyzer.analyze(text)
                results.append(analysis)
            except Exception as e:
                errors.append(str(e))
        
        if errors:
            return Result.failure(Exception(f"Batch analysis failed: {errors}"))
        
        return Result.success(results)