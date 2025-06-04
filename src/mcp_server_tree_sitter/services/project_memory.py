"""Project memory service for vector storage and retrieval using ChromaDB."""

import logging
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..bootstrap import get_logger
from .embedding_service import EmbeddingService, CodeChunk, EmbeddedChunk

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Represents a search result from memory."""
    
    chunk: CodeChunk
    similarity: float
    context: str
    
    def __post_init__(self):
        """Ensure similarity is between 0 and 1."""
        self.similarity = max(0.0, min(1.0, self.similarity))


class ProjectMemory:
    """Service for storing and retrieving code embeddings using ChromaDB."""
    
    def __init__(self, embedding_service: EmbeddingService, chroma_path: Optional[str] = None):
        """Initialize the project memory service.
        
        Args:
            embedding_service: The embedding service to use.
            chroma_path: Path to ChromaDB storage. If None, uses in-memory storage.
        """
        self.embedding_service = embedding_service
        self.chroma_path = chroma_path
        self._chroma_client = None
        self._collections: Dict[str, Any] = {}
        
        logger.info(f"Initialized ProjectMemory with chroma_path: {chroma_path}")
    
    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            try:
                import chromadb
                
                if self.chroma_path:
                    # Persistent storage
                    self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
                    logger.info(f"Created persistent ChromaDB client at {self.chroma_path}")
                else:
                    # In-memory storage for testing
                    self._chroma_client = chromadb.Client()
                    logger.info("Created in-memory ChromaDB client")
                    
            except ImportError:
                logger.error("ChromaDB not installed. Install with: pip install chromadb")
                raise
            except Exception as e:
                logger.error(f"Failed to create ChromaDB client: {e}")
                raise
        
        return self._chroma_client
    
    def _get_collection_name(self, project_name: str) -> str:
        """Generate a valid collection name for ChromaDB."""
        # ChromaDB collection names must be 3-63 characters, alphanumeric and hyphens
        # Replace invalid characters with hyphens
        valid_name = "".join(c if c.isalnum() else "-" for c in project_name)
        valid_name = valid_name.strip("-")
        
        # Ensure minimum length
        if len(valid_name) < 3:
            valid_name = f"project-{valid_name}"
        
        # Ensure maximum length
        if len(valid_name) > 63:
            valid_name = valid_name[:60] + "..."
        
        return valid_name
    
    def _get_or_create_collection(self, project_name: str):
        """Get or create a ChromaDB collection for a project."""
        collection_name = self._get_collection_name(project_name)
        
        if collection_name not in self._collections:
            try:
                client = self._get_chroma_client()
                
                # Try to get existing collection first
                try:
                    collection = client.get_collection(collection_name)
                    logger.debug(f"Retrieved existing collection: {collection_name}")
                except Exception:
                    # Collection doesn't exist, create it
                    collection = client.create_collection(
                        name=collection_name,
                        metadata={"project_name": project_name}
                    )
                    logger.info(f"Created new collection: {collection_name}")
                
                self._collections[collection_name] = collection
            except Exception as e:
                logger.error(f"Failed to get/create collection {collection_name}: {e}")
                raise
        
        return self._collections[collection_name]
    
    async def index_project(self, project_path: str, project_name: str, 
                          force_reindex: bool = False) -> Dict[str, Any]:
        """Index a project by extracting chunks and storing embeddings.
        
        Args:
            project_path: Path to the project directory.
            project_name: Name of the project.
            force_reindex: Whether to force re-indexing even if already indexed.
            
        Returns:
            Dictionary with indexing statistics.
        """
        logger.info(f"Starting indexing of project '{project_name}' at {project_path}")
        
        # Get or create collection
        collection = self._get_or_create_collection(project_name)
        
        # Check if already indexed (unless forcing reindex)
        if not force_reindex:
            try:
                count = collection.count()
                if count > 0:
                    logger.info(f"Project '{project_name}' already has {count} chunks indexed")
                    return {"status": "already_indexed", "chunk_count": count}
            except Exception as e:
                logger.warning(f"Failed to check existing index: {e}")
        
        # TODO: Integrate with existing tree-sitter parsing to extract chunks
        # For now, create a simple implementation that reads files
        chunks = await self._extract_chunks_from_project(project_path, project_name)
        
        if not chunks:
            logger.warning(f"No chunks extracted from project '{project_name}'")
            return {"status": "no_chunks", "chunk_count": 0}
        
        # Generate embeddings
        embedded_chunks = await self.embedding_service.batch_embed(chunks)
        
        # Store in ChromaDB
        stored_count = await self._store_embedded_chunks(collection, embedded_chunks)
        
        logger.info(f"Successfully indexed {stored_count} chunks for project '{project_name}'")
        
        return {
            "status": "indexed",
            "chunk_count": stored_count,
            "total_chunks_found": len(chunks),
            "project_path": project_path
        }
    
    async def query_memory(self, query: str, project_name: str, 
                          limit: int = 5, similarity_threshold: float = 0.0) -> List[RetrievalResult]:
        """Query the project memory for relevant code chunks.
        
        Args:
            query: The search query.
            project_name: The project to search in.
            limit: Maximum number of results to return.
            similarity_threshold: Minimum similarity score for results.
            
        Returns:
            List of RetrievalResult objects ordered by similarity.
        """
        logger.debug(f"Querying memory for project '{project_name}' with query: {query}")
        
        try:
            # Get collection
            collection = self._get_or_create_collection(project_name)
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_embedding(query)
            
            # Search in ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            
            if results['ids'] and len(results['ids']) > 0:
                ids = results['ids'][0]
                distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
                metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
                documents = results['documents'][0] if results['documents'] else [''] * len(ids)
                
                for i, (chunk_id, distance, metadata, document) in enumerate(zip(ids, distances, metadatas, documents)):
                    # Convert distance to similarity (ChromaDB returns distances, lower = more similar)
                    similarity = max(0.0, 1.0 - distance)
                    
                    if similarity >= similarity_threshold:
                        # Reconstruct CodeChunk from metadata
                        # Deserialize chunk metadata from JSON string
                        import json
                        try:
                            chunk_metadata = json.loads(metadata.get('chunk_metadata', '{}'))
                        except (json.JSONDecodeError, TypeError):
                            chunk_metadata = {}
                            
                        chunk = CodeChunk(
                            id=chunk_id,
                            content=document,
                            file_path=metadata.get('file_path', ''),
                            start_line=metadata.get('start_line', 0),
                            end_line=metadata.get('end_line', 0),
                            type=metadata.get('type', 'unknown'),
                            metadata=chunk_metadata
                        )
                        
                        # Create context (could be enhanced with surrounding code)
                        context = self._create_context(chunk, query)
                        
                        retrieval_results.append(RetrievalResult(
                            chunk=chunk,
                            similarity=similarity,
                            context=context
                        ))
            
            logger.debug(f"Found {len(retrieval_results)} results for query in project '{project_name}'")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed to query memory for project '{project_name}': {e}")
            return []
    
    async def update_chunks(self, project_name: str, changed_files: List[str]) -> Dict[str, Any]:
        """Incrementally update chunks for changed files.
        
        Args:
            project_name: The project name.
            changed_files: List of file paths that have changed.
            
        Returns:
            Dictionary with update statistics.
        """
        logger.info(f"Updating chunks for {len(changed_files)} changed files in project '{project_name}'")
        
        # Get collection
        collection = self._get_or_create_collection(project_name)
        
        updated_count = 0
        
        for file_path in changed_files:
            try:
                # Remove existing chunks for this file
                await self._remove_chunks_for_file(collection, file_path)
                
                # Extract new chunks for this file
                file_chunks = await self._extract_chunks_from_file(file_path, project_name)
                
                if file_chunks:
                    # Generate embeddings and store
                    embedded_chunks = await self.embedding_service.batch_embed(file_chunks)
                    stored = await self._store_embedded_chunks(collection, embedded_chunks)
                    updated_count += stored
                    logger.debug(f"Updated {stored} chunks for file {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to update chunks for file {file_path}: {e}")
                continue
        
        logger.info(f"Successfully updated {updated_count} chunks")
        return {"status": "updated", "updated_count": updated_count}
    
    async def _extract_chunks_from_project(self, project_path: str, project_name: str) -> List[CodeChunk]:
        """Extract code chunks from a project directory.
        
        This is a simplified implementation. In a full implementation,
        this would integrate with the existing tree-sitter parsing.
        """
        chunks = []
        project_dir = Path(project_path)
        
        if not project_dir.exists():
            logger.warning(f"Project path does not exist: {project_path}")
            return chunks
        
        # Simple file extension filter
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.go', '.rs', '.swift'}
        
        try:
            for file_path in project_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix in code_extensions:
                    file_chunks = await self._extract_chunks_from_file(str(file_path), project_name)
                    chunks.extend(file_chunks)
                    
                    # Limit for demo purposes
                    if len(chunks) > 100:
                        logger.warning(f"Limiting to first 100 chunks for demo")
                        break
        except Exception as e:
            logger.error(f"Failed to extract chunks from project {project_path}: {e}")
        
        logger.info(f"Extracted {len(chunks)} chunks from project '{project_name}'")
        return chunks
    
    async def _extract_chunks_from_file(self, file_path: str, project_name: str) -> List[CodeChunk]:
        """Extract code chunks from a single file."""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking strategy - split by functions/classes
            # In a full implementation, this would use tree-sitter AST
            lines = content.split('\n')
            
            current_chunk_lines = []
            current_start_line = 1
            chunk_id = 0
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Simple heuristic for chunk boundaries
                if (stripped.startswith('def ') or 
                    stripped.startswith('class ') or 
                    stripped.startswith('function ') or
                    stripped.startswith('export function')):
                    
                    # Save previous chunk if it exists
                    if current_chunk_lines:
                        chunk = self._create_chunk_from_lines(
                            current_chunk_lines, 
                            file_path, 
                            project_name,
                            current_start_line,
                            i - 1,
                            chunk_id
                        )
                        if chunk:
                            chunks.append(chunk)
                        chunk_id += 1
                    
                    # Start new chunk
                    current_chunk_lines = [line]
                    current_start_line = i
                else:
                    current_chunk_lines.append(line)
            
            # Don't forget the last chunk
            if current_chunk_lines:
                chunk = self._create_chunk_from_lines(
                    current_chunk_lines,
                    file_path,
                    project_name, 
                    current_start_line,
                    len(lines),
                    chunk_id
                )
                if chunk:
                    chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Failed to extract chunks from file {file_path}: {e}")
        
        return chunks
    
    def _create_chunk_from_lines(self, lines: List[str], file_path: str, project_name: str,
                                start_line: int, end_line: int, chunk_id: int) -> Optional[CodeChunk]:
        """Create a CodeChunk from lines of code."""
        content = '\n'.join(lines).strip()
        
        if not content or len(content) < 10:  # Skip very small chunks
            return None
        
        # Simple type detection
        first_line = lines[0].strip() if lines else ""
        if first_line.startswith('class '):
            chunk_type = 'class'
        elif first_line.startswith('def ') or first_line.startswith('function '):
            chunk_type = 'function'
        else:
            chunk_type = 'module'
        
        # Generate unique ID
        chunk_id_str = f"{project_name}_{Path(file_path).name}_{start_line}_{chunk_id}"
        
        return CodeChunk(
            id=chunk_id_str,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            type=chunk_type,
            metadata={
                'project_name': project_name,
                'lines_count': len(lines)
            }
        )
    
    async def _store_embedded_chunks(self, collection, embedded_chunks: List[EmbeddedChunk]) -> int:
        """Store embedded chunks in ChromaDB collection."""
        if not embedded_chunks:
            return 0
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.id for chunk in embedded_chunks]
            embeddings = [chunk.embedding for chunk in embedded_chunks]
            documents = [chunk.content for chunk in embedded_chunks]
            metadatas = []
            
            for chunk in embedded_chunks:
                # Convert chunk metadata to JSON string for ChromaDB compatibility
                import json
                chunk_metadata_str = json.dumps(chunk.metadata) if chunk.metadata else "{}"
                
                metadata = {
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'type': chunk.type,
                    'hash': chunk.hash,
                    'chunk_metadata': chunk_metadata_str
                }
                metadatas.append(metadata)
            
            # Store in ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.debug(f"Stored {len(embedded_chunks)} chunks in ChromaDB")
            return len(embedded_chunks)
            
        except Exception as e:
            logger.error(f"Failed to store embedded chunks: {e}")
            return 0
    
    async def _remove_chunks_for_file(self, collection, file_path: str):
        """Remove all chunks for a specific file from the collection."""
        try:
            # Query for chunks from this file
            results = collection.get(
                where={"file_path": file_path}
            )
            
            if results['ids']:
                collection.delete(ids=results['ids'])
                logger.debug(f"Removed {len(results['ids'])} chunks for file {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to remove chunks for file {file_path}: {e}")
    
    def _create_context(self, chunk: CodeChunk, query: str) -> str:
        """Create context string for a retrieval result."""
        context_parts = [
            f"File: {chunk.file_path}",
            f"Lines: {chunk.start_line}-{chunk.end_line}",
            f"Type: {chunk.type}"
        ]
        
        if chunk.metadata.get('project_name'):
            context_parts.append(f"Project: {chunk.metadata['project_name']}")
        
        return " | ".join(context_parts)
    
    def get_project_stats(self, project_name: str) -> Dict[str, Any]:
        """Get statistics for a project's memory."""
        try:
            collection = self._get_or_create_collection(project_name)
            count = collection.count()
            
            return {
                "project_name": project_name,
                "chunk_count": count,
                "collection_name": self._get_collection_name(project_name)
            }
        except Exception as e:
            logger.error(f"Failed to get stats for project '{project_name}': {e}")
            return {"project_name": project_name, "chunk_count": 0, "error": str(e)}
