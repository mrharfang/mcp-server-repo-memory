"""Memory-related MCP tools for project memory and semantic search."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


from concurrent.futures import ThreadPoolExecutor
import threading


def _run_async_in_sync(async_func, *args, **kwargs):
    """Run an async function from sync context safely.
    
    This avoids the 'asyncio.run() cannot be called from a running event loop' error
    by running the async function in a separate thread with its own event loop.
    """
    def run_in_thread():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
    
    # Run the async function in a separate thread
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result()


def register_memory_tools(mcp_server: Any, project_memory: Any, embedding_service: Any) -> None:
    """Register memory-related tools with the MCP server.
    
    Args:
        mcp_server: MCP server instance
        project_memory: ProjectMemory service instance
        embedding_service: EmbeddingService instance
    """
    logger = logging.getLogger(__name__)

    @mcp_server.tool()
    def index_project_memory(
        project_path: str,
        project_name: str,
        force_reindex: bool = False,
    ) -> str:
        """Create or update semantic memory index for a codebase project.

        Args:
            project_path: Absolute path to project root directory
            project_name: Unique identifier for this project
            force_reindex: Force complete reindexing even if up to date

        Returns:
            JSON string with indexing results
        """
        # Validate project path exists
        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            return json.dumps({
                "success": False,
                "error": f"Project path does not exist: {project_path}"
            })

        start_time = time.time()

        try:
            # Call ProjectMemory.index_project() (async method)
            # Use thread-based async execution to avoid event loop conflicts
            result = _run_async_in_sync(
                project_memory.index_project,
                project_path=project_path,
                project_name=project_name,
                force_reindex=force_reindex
            )

            elapsed_time = time.time() - start_time

            response = {
                "success": True,
                "project_name": project_name,
                "chunks_indexed": result.get("chunks_indexed", 0),
                "files_processed": result.get("files_processed", 0),
                "time_taken": f"{elapsed_time:.2f}s",
                "index_size_mb": result.get("index_size_mb", 0)
            }

            return json.dumps(response, indent=2)

        except Exception as e:
            logger.error(f"Error indexing project: {e}")
            return json.dumps({
                "success": False,
                "error": f"Error indexing project: {str(e)}"
            })

    @mcp_server.tool()
    def query_project_memory(
        project_name: str,
        query: str,
        context_limit: int = 5,
        include_metadata: bool = True,
    ) -> str:
        """Semantic search across indexed codebase using natural language.

        Args:
            project_name: Name of previously indexed project
            query: Natural language query about the codebase
            context_limit: Maximum number of relevant code chunks to return
            include_metadata: Include file paths, line numbers, and similarity scores

        Returns:
            Formatted text with search results
        """
        start_time = time.time()

        try:
            # Call ProjectMemory.query_memory() (async method)
            # Use thread-based async execution to avoid event loop conflicts
            results = _run_async_in_sync(
                project_memory.query_memory,
                query=query,
                project_name=project_name,
                limit=context_limit
            )

            execution_time = (time.time() - start_time) * 1000  # ms

            if not results:
                return f"No relevant results found for query: '{query}' in project '{project_name}'"

            # Format results for display
            response_text = f"Query: '{query}'\nProject: '{project_name}'\n\n"
            response_text += f"Found {len(results)} relevant code chunks:\n\n"

            for i, result in enumerate(results, 1):
                chunk = result.chunk
                similarity = result.similarity

                response_text += f"## Result {i} (Similarity: {similarity:.3f})\n"

                if include_metadata:
                    response_text += f"**File:** {chunk.file_path}\n"
                    response_text += f"**Lines:** {chunk.start_line}-{chunk.end_line}\n"
                    response_text += f"**Type:** {chunk.type}\n\n"

                # Truncate very long code chunks
                content = chunk.content
                if len(content) > 500:
                    content = content[:500] + "...\n[truncated]"

                response_text += f"```{getattr(chunk, 'language', '')}\n"
                response_text += f"{content}\n```\n\n"

            response_text += f"Query executed in {execution_time:.1f}ms"

            return response_text

        except Exception as e:
            logger.error(f"Error querying project memory: {e}")
            return f"Error querying project memory: {str(e)}"

    @mcp_server.tool()
    def list_project_memories(
        detailed: bool = False,
    ) -> str:
        """Show all indexed projects and their memory statistics.

        Args:
            detailed: Include detailed statistics for each project

        Returns:
            Formatted text with project memory information
        """
        try:
            # Access ChromaDB client directly to list collections
            client = project_memory._get_chroma_client()
            collections = client.list_collections()

            if not collections:
                return "No indexed projects found."

            response_text = f"Found {len(collections)} indexed project(s):\n\n"
            total_chunks = 0

            for collection in collections:
                collection_name = collection.name
                chunk_count = collection.count()
                total_chunks += chunk_count

                response_text += f"## {collection_name}\n"
                response_text += f"**Collection Name:** {collection_name}\n"
                response_text += f"**Code Chunks:** {chunk_count}\n"

                if detailed:
                    # Get some sample data if detailed is requested
                    try:
                        if chunk_count > 0:
                            sample = collection.peek(limit=1)
                            if sample and sample.get('metadatas'):
                                metadata = sample['metadatas'][0]
                                if 'file_path' in metadata:
                                    response_text += f"**Sample File:** {metadata['file_path']}\n"
                                if 'project_name' in metadata:
                                    response_text += f"**Project Name:** {metadata['project_name']}\n"
                    except Exception:
                        pass  # Skip detailed info if we can't get it

                response_text += "\n"

            # Summary
            response_text += "---\n"
            response_text += f"**Total Projects:** {len(collections)}\n"
            response_text += f"**Total Code Chunks:** {total_chunks}\n"

            return response_text

        except Exception as e:
            logger.error(f"Error listing project memories: {e}")
            return f"Error listing project memories: {str(e)}"
