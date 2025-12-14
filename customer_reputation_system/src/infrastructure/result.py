"""Result pattern for consistent error handling."""

from typing import Generic, TypeVar, Any, Optional, Callable
try:
    from .exceptions import ReputationSystemException, ValidationError
except ImportError:
    # Fallback definitions for when exceptions module isn't available
    class ReputationSystemException(Exception):
        pass
    
    class ValidationError(ReputationSystemException):
        pass

T = TypeVar('T')


class Result(Generic[T]):
    """Result pattern for consistent error handling."""
    
    def __init__(self, value: Optional[T] = None, error: Optional[Exception] = None):
        self.value = value
        self.error = error
    
    @property
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self.error is None
    
    @property
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return self.error is not None
    
    @classmethod
    def success(cls, value: T) -> 'Result[T]':
        """Create a successful result."""
        return cls(value=value, error=None)
    
    @classmethod
    def failure(cls, error: Exception) -> 'Result[T]':
        """Create a failure result."""
        return cls(value=None, error=error)
    
    def map(self, func: Callable[[T], Any]) -> 'Result[Any]':
        """Map function over successful value."""
        if self.is_success and self.value is not None:
            try:
                return Result.success(func(self.value))
            except Exception as e:
                return Result.failure(e)
        if self.error is not None:
            return Result.failure(self.error)
        return Result.failure(Exception("Unknown error"))
    
    def flat_map(self, func: Callable[[T], Any]) -> Any:
        """Flat map function over successful value."""
        if self.is_success and self.value is not None:
            try:
                result = func(self.value)
                if hasattr(result, 'is_success'):
                    return result
                return Result.success(result)
            except Exception as e:
                return Result.failure(e)
        if self.error is not None:
            return Result.failure(self.error)
        return Result.failure(Exception("Unknown error"))
    
    def on_success(self, func: Callable[[T], None]) -> 'Result[T]':
        """Execute function on success, return original result."""
        if self.is_success and self.value is not None:
            func(self.value)
        return self
    
    def on_failure(self, func: Callable[[Exception], None]) -> 'Result[T]':
        """Execute function on failure, return original result."""
        if self.is_failure and self.error is not None:
            func(self.error)
        return self
    
    def get_or_else(self, default_value: T) -> T:
        """Get value or return default."""
        return self.value if self.is_success else default_value
    
    def get_or_raise(self) -> T:
        """Get value or raise exception."""
        if self.is_success:
            assert self.value is not None, "Success result has no value"
            return self.value
        if self.error is not None:
            raise self.error
        raise ValueError("Result has no value and no error")
    
    def __str__(self) -> str:
        if self.is_success:
            return f"Success({self.value})"
        return f"Failure({self.error})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ResultBuilder:
    """Builder for creating results with validation."""
    
    @staticmethod
    def validate(condition: bool, error_message: str) -> Result[None]:
        """Create result based on condition."""
        if condition:
            return Result.success(None)
        return Result.failure(ValidationError(error_message))
    
    @staticmethod
    def try_execute(func: Callable, *args, **kwargs) -> Result[Any]:
        """Execute function and catch exceptions."""
        try:
            return Result.success(func(*args, **kwargs))
        except ReputationSystemException:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            return Result.failure(e)