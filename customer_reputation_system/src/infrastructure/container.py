"""Service container for dependency injection."""

from typing import Dict, Callable, Any, Optional
import logging


class ServiceContainer:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_flags: Dict[str, bool] = {}
        self._logger = logging.getLogger(__name__)
    
    def register(self, name: str, factory: Callable, singleton: bool = True):
        """Register a service factory.
        
        Args:
            name: Service name
            factory: Function that creates the service
            singleton: Whether to cache the service instance
        """
        self._factories[name] = factory
        self._singleton_flags[name] = singleton
        self._logger.debug(f"Registered service: {name}")
    
    def get(self, name: str) -> Any:
        """Get a service instance.
        
        Args:
            name: Service name
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service not registered
        """
        if name not in self._factories:
            raise ValueError(f"Service '{name}' not registered")
        
        # Return singleton if already created
        if self._singleton_flags.get(name, True) and name in self._singletons:
            return self._singletons[name]
        
        # Create new instance
        factory = self._factories[name]
        instance = factory()
        
        # Cache singleton
        if self._singleton_flags.get(name, True):
            self._singletons[name] = instance
        
        self._logger.debug(f"Created service instance: {name}")
        return instance
    
    def has(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._factories
    
    def clear(self):
        """Clear all singletons (useful for testing)."""
        self._singletons.clear()
        self._logger.debug("Cleared all singleton services")


# Global container instance
container = ServiceContainer()