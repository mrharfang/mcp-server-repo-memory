"""Dependency injection container for MCP Tree-sitter Server.

This module provides a central container for managing all application dependencies,
replacing the global variables and singletons previously used throughout the codebase.
"""

from typing import Any, Dict, Optional

# Import logging from bootstrap package
from .bootstrap import get_logger
from .cache.parser_cache import TreeCache
from .config import ConfigurationManager, ServerConfig
from .language.registry import LanguageRegistry
from .models.project import ProjectRegistry

logger = get_logger(__name__)


class DependencyContainer:
    """Container for all application dependencies."""

    def __init__(self) -> None:
        """Initialize container with all core dependencies."""
        logger.debug("Initializing dependency container")

        # Create core dependencies
        self.config_manager = ConfigurationManager()
        self._config = self.config_manager.get_config()
        self.project_registry = ProjectRegistry()
        self.language_registry = LanguageRegistry()
        self.tree_cache = TreeCache(
            max_size_mb=self._config.cache.max_size_mb, ttl_seconds=self._config.cache.ttl_seconds
        )

        # Memory services (lazy initialization)
        self._embedding_service: Optional[Any] = None
        self._project_memory: Optional[Any] = None

        # Storage for any additional dependencies
        self._additional: Dict[str, Any] = {}

    def get_config(self) -> ServerConfig:
        """Get the current configuration."""
        # Always get the latest from the config manager
        config = self.config_manager.get_config()
        return config

    def get_embedding_service(self):
        """Get or create the embedding service."""
        if self._embedding_service is None:
            from .services.embedding_service import EmbeddingService
            self._embedding_service = EmbeddingService()
            logger.debug("Created EmbeddingService instance")
        return self._embedding_service

    def get_project_memory(self):
        """Get or create the project memory service."""
        if self._project_memory is None:
            from .services.project_memory import ProjectMemory
            config = self.get_config()
            embedding_service = self.get_embedding_service()
            self._project_memory = ProjectMemory(
                embedding_service=embedding_service,
                chroma_path=config.memory.chroma_path
            )
            logger.debug("Created ProjectMemory instance")
        return self._project_memory

    def register_dependency(self, name: str, instance: Any) -> None:
        """Register an additional dependency."""
        self._additional[name] = instance

    def get_dependency(self, name: str) -> Any:
        """Get a registered dependency."""
        return self._additional.get(name)

    @property
    def embedding_service(self):
        """Property accessor for embedding service."""
        return self.get_embedding_service()

    @property
    def project_memory(self):
        """Property accessor for project memory."""
        return self.get_project_memory()


# Create the single container instance - this will be the ONLY global
container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get the dependency container."""
    return container
