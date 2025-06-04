"""Embedding service for generating vector embeddings from code chunks."""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..bootstrap import get_logger

logger = get_logger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    type: str  # 'function', 'class', 'method', 'module'
    metadata: Dict[str, Any]


@dataclass
class EmbeddedChunk:
    """Represents a code chunk with its embedding vector."""
    
    chunk: CodeChunk
    embedding: List[float]
    hash: str
    
    @property
    def id(self) -> str:
        """Get the chunk ID."""
        return self.chunk.id
    
    @property
    def content(self) -> str:
        """Get the chunk content."""
        return self.chunk.content
    
    @property
    def file_path(self) -> str:
        """Get the file path."""
        return self.chunk.file_path
    
    @property
    def start_line(self) -> int:
        """Get the start line."""
        return self.chunk.start_line
    
    @property
    def end_line(self) -> int:
        """Get the end line."""
        return self.chunk.end_line
    
    @property
    def type(self) -> str:
        """Get the chunk type."""
        return self.chunk.type
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the metadata."""
        return self.chunk.metadata


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        pass


class MockEmbeddingBackend(EmbeddingBackend):
    """Mock embedding backend for testing and development."""
    
    def __init__(self, dimension: int = 384):
        """Initialize with specified dimension."""
        self.dimension = dimension
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate a mock embedding vector based on text hash."""
        # Create a deterministic "embedding" based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to integers and normalize
        hash_ints = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
        
        # Extend or truncate to desired dimension
        if len(hash_ints) < self.dimension:
            # Repeat the pattern to fill dimension
            multiplier = (self.dimension // len(hash_ints)) + 1
            hash_ints = (hash_ints * multiplier)[:self.dimension]
        else:
            hash_ints = hash_ints[:self.dimension]
        
        # Normalize to [-1, 1] range
        normalized = [(x - 127.5) / 127.5 for x in hash_ints]
        return normalized
    
    def get_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        return self.dimension


class EmbeddingService:
    """Service for generating embeddings from code chunks."""
    
    def __init__(self, backend: Optional[EmbeddingBackend] = None):
        """Initialize the embedding service.
        
        Args:
            backend: The embedding backend to use. If None, uses MockEmbeddingBackend.
        """
        self.backend = backend or MockEmbeddingBackend()
        logger.info(f"Initialized EmbeddingService with {type(self.backend).__name__}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text.
        
        Args:
            text: The text to generate an embedding for.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        try:
            embedding = await self.backend.generate_embedding(text)
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text of length {len(text)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _create_content_hash(self, content: str) -> str:
        """Create a hash of the content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def embed_chunk(self, chunk: CodeChunk) -> EmbeddedChunk:
        """Generate an embedding for a single code chunk.
        
        Args:
            chunk: The code chunk to embed.
            
        Returns:
            An EmbeddedChunk with the generated embedding.
        """
        # Prepare text for embedding (could include metadata context)
        text_for_embedding = self._prepare_text_for_embedding(chunk)
        
        # Generate embedding
        embedding = await self.generate_embedding(text_for_embedding)
        
        # Create content hash
        content_hash = self._create_content_hash(chunk.content)
        
        return EmbeddedChunk(
            chunk=chunk,
            embedding=embedding,
            hash=content_hash
        )
    
    async def batch_embed(self, chunks: List[CodeChunk], batch_size: int = 10) -> List[EmbeddedChunk]:
        """Process chunks in batches to generate embeddings.
        
        Args:
            chunks: List of code chunks to embed.
            batch_size: Number of chunks to process in each batch.
            
        Returns:
            List of EmbeddedChunk objects with generated embeddings.
        """
        logger.info(f"Starting batch embedding of {len(chunks)} chunks with batch size {batch_size}")
        
        embedded_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}: chunks {i} to {i + len(batch) - 1}")
            
            # Process each chunk in the batch
            for chunk in batch:
                try:
                    embedded_chunk = await self.embed_chunk(chunk)
                    embedded_chunks.append(embedded_chunk)
                except Exception as e:
                    logger.error(f"Failed to embed chunk {chunk.id}: {e}")
                    # Continue with other chunks
                    continue
        
        logger.info(f"Successfully embedded {len(embedded_chunks)} out of {len(chunks)} chunks")
        return embedded_chunks
    
    def _prepare_text_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare text for embedding by including relevant context.
        
        Args:
            chunk: The code chunk to prepare.
            
        Returns:
            The prepared text for embedding.
        """
        # Start with the content
        text_parts = [chunk.content]
        
        # Add type information
        text_parts.append(f"Type: {chunk.type}")
        
        # Add file path context (just filename for now)
        from pathlib import Path
        filename = Path(chunk.file_path).name
        text_parts.append(f"File: {filename}")
        
        # Add any relevant metadata
        if chunk.metadata:
            if 'name' in chunk.metadata:
                text_parts.append(f"Name: {chunk.metadata['name']}")
            if 'docstring' in chunk.metadata:
                text_parts.append(f"Documentation: {chunk.metadata['docstring']}")
        
        return "\n".join(text_parts)
    
    def get_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors."""
        return self.backend.get_dimension()
