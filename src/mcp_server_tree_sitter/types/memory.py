"""Type definitions for memory functionality."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Configuration for memory functionality."""
    
    chroma_path: str = Field(default="./chroma_db", description="Path to ChromaDB storage")
    embedding_model: str = Field(default="mock", description="Embedding model to use")
    max_chunk_size: int = Field(default=1000, description="Maximum size of code chunks")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity for search results")
    max_results_default: int = Field(default=10, description="Default maximum results for queries")


@dataclass
class ProjectIndex:
    """Information about a project's memory index."""
    
    name: str
    path: str
    last_indexed: datetime
    chunk_count: int
    file_count: int
    embedding_dimension: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "path": self.path,
            "last_indexed": self.last_indexed.isoformat(),
            "chunk_count": self.chunk_count,
            "file_count": self.file_count,
            "embedding_dimension": self.embedding_dimension
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectIndex":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            path=data["path"],
            last_indexed=datetime.fromisoformat(data["last_indexed"]),
            chunk_count=data["chunk_count"],
            file_count=data["file_count"],
            embedding_dimension=data["embedding_dimension"]
        )


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    
    total_projects: int
    total_chunks: int
    total_size_mb: float
    indexed_files: int
    embedding_dimension: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_projects": self.total_projects,
            "total_chunks": self.total_chunks,
            "total_size_mb": self.total_size_mb,
            "indexed_files": self.indexed_files,
            "embedding_dimension": self.embedding_dimension
        }
