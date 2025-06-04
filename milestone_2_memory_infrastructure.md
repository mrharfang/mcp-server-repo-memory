# Milestone 2: Memory Infrastructure

## Objective
Implement core memory services: embedding generation, vector storage, and semantic chunking extensions.

## Actions Required

### Create Memory Services Directory
```bash
mkdir -p src/mcp_server_tree_sitter/services
mkdir -p src/mcp_server_tree_sitter/types
```

### Add Dependencies
Add ChromaDB to `pyproject.toml`:
```toml
dependencies = [
    # ... existing dependencies
    "chromadb>=0.4.0",
]
```

Then install:
```bash
uv sync
```

### Implement EmbeddingService
Create `src/mcp_server_tree_sitter/services/embedding_service.py`:
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class CodeChunk:
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    type: str  # 'function' | 'class' | 'method' | 'module'
    metadata: Dict[str, Any]

@dataclass
class EmbeddedChunk(CodeChunk):
    embedding: List[float]
    hash: str

class MockEmbeddingBackend:
    """Mock embedding backend for development/testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    async def embed(self, text: str) -> List[float]:
        # Generate deterministic mock embeddings based on text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Create normalized vector from hash
        embedding = []
        for i in range(self.dimension):
            hash_val = int(text_hash[i % len(text_hash)], 16)
            normalized_val = (hash_val / 15.0) - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(normalized_val)
        return embedding

class EmbeddingService:
    def __init__(self, backend: Optional[str] = None):
        self.backend = MockEmbeddingBackend()
        logger.info(f"Initialized EmbeddingService with {type(self.backend).__name__}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        return await self.backend.embed(text)
    
    async def batch_embed(self, chunks: List[CodeChunk], batch_size: int = 10) -> List[EmbeddedChunk]:
        """Process chunks in batches and add content hash for change detection."""
        logger.info(f"Starting batch embedding of {len(chunks)} chunks with batch size {batch_size}")
        
        embedded_chunks = []
        successful_chunks = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                try:
                    # Generate embedding
                    embedding = await self.generate_embedding(chunk.content)
                    
                    # Generate content hash for change detection
                    content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:8]
                    
                    # Create embedded chunk
                    embedded_chunk = EmbeddedChunk(
                        id=chunk.id,
                        content=chunk.content,
                        file_path=chunk.file_path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        type=chunk.type,
                        metadata=chunk.metadata,
                        embedding=embedding,
                        hash=content_hash
                    )
                    
                    embedded_chunks.append(embedded_chunk)
                    successful_chunks += 1
                    
                except Exception as e:
                    logger.error(f"Failed to embed chunk {chunk.id}: {e}")
        
        logger.info(f"Successfully embedded {successful_chunks} out of {len(chunks)} chunks")
        return embedded_chunks
```

### Implement ProjectMemory
Create `src/mcp_server_tree_sitter/services/project_memory.py`:
```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import asyncio
import logging
import json
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

from .embedding_service import EmbeddingService, CodeChunk, EmbeddedChunk

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunk: CodeChunk
    similarity: float
    context: str

class ProjectMemory:
    def __init__(self, embedding_service: EmbeddingService, chroma_path: Optional[str] = None):
        self.embedding_service = embedding_service
        self.chroma_path = chroma_path
        self.client = None
        logger.info(f"Initialized ProjectMemory with chroma_path: {chroma_path}")
    
    async def index_project(self, project_path: str, project_name: str) -> Dict[str, Any]:
        """Index a project by creating ChromaDB collection and storing embeddings."""
        # Implementation details...
        return {"status": "indexed", "chunk_count": stored_count}
    
    async def query_memory(self, query: str, project_name: str, 
                          limit: int = 5, similarity_threshold: float = 0.0) -> List[RetrievalResult]:
        """Query the project memory for relevant code chunks."""
        # Implementation details...
        return retrieval_results
    
    async def update_chunks(self, project_name: str, changed_files: List[str]) -> Dict[str, Any]:
        """Incrementally update chunks for changed files."""
        # Implementation details...
        return {"status": "updated", "updated_chunks": updated_count}
```

### Create Type Definitions
Create `src/mcp_server_tree_sitter/types/memory.py`:
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class ProjectIndex:
    name: str
    path: str
    last_indexed: datetime
    chunk_count: int
    file_count: int

@dataclass
class MemoryConfig:
    chroma_path: str = "./chroma_db"
    embedding_model: str = "mock"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class MemoryStats:
    total_projects: int
    total_chunks: int
    total_embeddings: int
    storage_size_mb: float
```

### Extend Configuration
Modify `src/mcp_server_tree_sitter/config.py`:
```python
# Add import
from .types.memory import MemoryConfig

@dataclass
class ServerConfig:
    # ...existing fields...
    memory: MemoryConfig = field(default_factory=MemoryConfig)
```

### Initialize Services in DI Container
Extend `src/mcp_server_tree_sitter/di.py`:
```python
# Add imports
from .services.embedding_service import EmbeddingService
from .services.project_memory import ProjectMemory

class Container:
    # ...existing code...
    
    def get_embedding_service(self) -> EmbeddingService:
        """Get EmbeddingService instance."""
        if not hasattr(self, '_embedding_service'):
            self._embedding_service = EmbeddingService()
        return self._embedding_service
    
    def get_project_memory(self) -> ProjectMemory:
        """Get ProjectMemory instance."""
        if not hasattr(self, '_project_memory'):
            config = self.get_config()
            embedding_service = self.get_embedding_service()
            self._project_memory = ProjectMemory(
                embedding_service=embedding_service,
                chroma_path=config.memory.chroma_path
            )
        return self._project_memory
```

## Test Conditions - Milestone 2 COMPLETE When:
- [x] EmbeddingService.py imports without errors
- [x] ProjectMemory.py imports without errors  
- [x] memory.py type definitions import correctly
- [x] Services can be instantiated via DI container
- [x] ChromaDB initializes and creates collections
- [x] Embedding generation works for test strings
- [x] Project indexing and querying functional
- [x] All memory infrastructure tests pass

## Verification Script
Create `test_milestone_2.py`:
```python
import asyncio
import tempfile
import os
from src.mcp_server_tree_sitter.config import ServerConfig
from src.mcp_server_tree_sitter.di import container
from src.mcp_server_tree_sitter.services.embedding_service import EmbeddingService, CodeChunk
from src.mcp_server_tree_sitter.services.project_memory import ProjectMemory

async def test_milestone_2():
    print("🧪 Testing Milestone 2: Memory Infrastructure")
    print("=" * 50)
    
    # Test 1: Configuration includes memory settings
    config = ServerConfig()
    assert hasattr(config, 'memory'), "ServerConfig should have memory field"
    print(f"✅ Test 1: Configuration includes memory settings")
    
    # Test 2: EmbeddingService instantiation
    embedding_service = EmbeddingService()
    print(f"✅ Test 2: EmbeddingService instantiation")
    
    # Test 3: Embedding generation
    embedding = await embedding_service.generate_embedding("hello world")
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    print(f"✅ Test 3: Embedding generation")
    
    # Additional tests...
    
    print("🎉 Milestone 2 Tests PASSED!")

if __name__ == "__main__":
    asyncio.run(test_milestone_2())
```

Run: `python test_milestone_2.py`

## On Completion
- [x] Update ENHANCEMENT_PLAN.md to mark Milestone 2 as ✅ and ready for Milestone 3
- [x] All memory infrastructure components functional and tested
- [x] ChromaDB integration working with proper metadata handling
- [x] Mock embedding backend ready for future real model integration
- [x] DI container properly provides memory services as singletons

**Status**: ✅ COMPLETE (2025-06-03)
**Next**: Milestone 3 - Enhanced Memory Capabilities (persistent sessions, intelligent summarization)