"""Custom exception hierarchy for reputation system."""

from typing import Optional


class ReputationSystemException(Exception):
    """Base exception for reputation system."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class ValidationError(ReputationSystemException):
    """Data validation failed."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field


class DatabaseError(ReputationSystemException):
    """Database operation failed."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, "DATABASE_ERROR")
        self.operation = operation


class ServiceUnavailableError(ReputationSystemException):
    """Optional service not available."""
    
    def __init__(self, message: str, service_name: Optional[str] = None):
        super().__init__(message, "SERVICE_UNAVAILABLE")
        self.service_name = service_name


class ConfigurationError(ReputationSystemException):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIG_ERROR")
        self.config_key = config_key


class NotFoundError(ReputationSystemException):
    """Resource not found."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        super().__init__(message, "NOT_FOUND")
        self.resource_type = resource_type
        self.resource_id = resource_id


class PermissionError(ReputationSystemException):
    """Permission denied."""
    
    def __init__(self, message: str, action: Optional[str] = None):
        super().__init__(message, "PERMISSION_ERROR")
        self.action = action


class BusinessLogicError(ReputationSystemException):
    """Business logic violation."""
    
    def __init__(self, message: str, rule: Optional[str] = None):
        super().__init__(message, "BUSINESS_LOGIC_ERROR")
        self.rule = rule