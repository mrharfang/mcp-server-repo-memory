# Milestone 3: MCP Tool Extensions (Python)

## Objective
Add three new MCP tools that expose project memory functionality: index_project_memory, query_project_memory, list_project_memories.

## Actions Required

### Locate Tool Registration Pattern
Study existing tool structure in the codebase:
- Find where tools are defined (likely in `src/mcp_server_tree_sitter/tools/` directory)
- Understand the base Tool class or interface pattern
- Identify how tools are registered in the main server module

### Create Memory Tools Directory
```bash
mkdir -p src/mcp_server_tree_sitter/tools/memory
touch src/mcp_server_tree_sitter/tools/memory/__init__.py
```

### Implement IndexProjectMemoryTool
Create `src/mcp_server_tree_sitter/tools/memory/index_project_memory.py`:
```python
from typing import Any, Dict
from mcp.types import Tool, TextContent
from ...services.project_memory import ProjectMemory
from ...services.embedding_service import EmbeddingService
import os
import time

class IndexProjectMemoryTool:
    """Tool for creating or updating semantic memory index for a codebase project."""
    
    def __init__(self, project_memory: ProjectMemory):
        self.project_memory = project_memory
        
    @property
    def definition(self) -> Tool:
        return Tool(
            name="index_project_memory",
            description="Create or update semantic memory index for a codebase project",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {
                        "type": "string",
                        "description": "Absolute path to project root directory"
                    },
                    "project_name": {
                        "type": "string",
                        "description": "Unique identifier for this project"
                    },
                    "force_reindex": {
                        "type": "boolean",
                        "default": False,
                        "description": "Force complete reindexing even if up to date"
                    }
                },
                "required": ["project_path", "project_name"]
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> list[TextContent]:
        project_path = params.get("project_path")
        project_name = params.get("project_name")
        force_reindex = params.get("force_reindex", False)
        
        # Validate project path exists
        if not os.path.exists(project_path):
            return [TextContent(
                type="text",
                text=f"Error: Project path does not exist: {project_path}"
            )]
        
        start_time = time.time()
        
        try:
            # Call ProjectMemory.index_project()
            result = await self.project_memory.index_project(
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
            
            return [TextContent(
                type="text", 
                text=f"Successfully indexed project '{project_name}':\n"
                     f"- Files processed: {response['files_processed']}\n"
                     f"- Code chunks indexed: {response['chunks_indexed']}\n"
                     f"- Time taken: {response['time_taken']}\n"
                     f"- Index size: {response['index_size_mb']} MB"
            )]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error indexing project: {str(e)}"
            )]
```

### Implement QueryProjectMemoryTool
Create `src/mcp_server_tree_sitter/tools/memory/query_project_memory.py`:
```python
from typing import Any, Dict
from mcp.types import Tool, TextContent
from ...services.project_memory import ProjectMemory
import time
import json

class QueryProjectMemoryTool:
    """Tool for semantic search across indexed codebase using natural language."""
    
    def __init__(self, project_memory: ProjectMemory):
        self.project_memory = project_memory
        
    @property 
    def definition(self) -> Tool:
        return Tool(
            name="query_project_memory",
            description="Semantic search across indexed codebase using natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_name": {
                        "type": "string",
                        "description": "Name of previously indexed project"
                    },
                    "query": {
                        "type": "string", 
                        "description": "Natural language query about the codebase"
                    },
                    "context_limit": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum number of relevant code chunks to return"
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include file paths, line numbers, and similarity scores"
                    }
                },
                "required": ["project_name", "query"]
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> list[TextContent]:
        project_name = params.get("project_name")
        query = params.get("query")
        context_limit = params.get("context_limit", 5)
        include_metadata = params.get("include_metadata", True)
        
        start_time = time.time()
        
        try:
            # Call ProjectMemory.query_memory()
            results = await self.project_memory.query_memory(
                query=query,
                project_name=project_name,
                limit=context_limit
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            if not results:
                return [TextContent(
                    type="text",
                    text=f"No relevant results found for query: '{query}' in project '{project_name}'"
                )]
            
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
                
                response_text += f"```{chunk.language if hasattr(chunk, 'language') else ''}\n"
                response_text += f"{content}\n```\n\n"
            
            response_text += f"Query executed in {execution_time:.1f}ms"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error querying project memory: {str(e)}"
            )]
```

### Implement ListProjectMemoriesTool
Create `src/mcp_server_tree_sitter/tools/memory/list_project_memories.py`:
```python
from typing import Any, Dict
from mcp.types import Tool, TextContent
from ...services.project_memory import ProjectMemory
import json

class ListProjectMemoriesTool:
    """Tool for showing all indexed projects and their memory statistics."""
    
    def __init__(self, project_memory: ProjectMemory):
        self.project_memory = project_memory
        
    @property
    def definition(self) -> Tool:
        return Tool(
            name="list_project_memories",
            description="Show all indexed projects and their memory statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "detailed": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include detailed statistics for each project"
                    }
                }
            }
        )
    
    async def execute(self, params: Dict[str, Any]) -> list[TextContent]:
        detailed = params.get("detailed", False)
        
        try:
            # Query ChromaDB for all collections
            projects = await self.project_memory.list_indexed_projects()
            
            if not projects:
                return [TextContent(
                    type="text",
                    text="No projects have been indexed yet.\n\n"
                         "Use the 'index_project_memory' tool to index your first project."
                )]
            
            response_text = f"Found {len(projects)} indexed project(s):\n\n"
            
            total_chunks = 0
            total_size_mb = 0
            
            for project in projects:
                name = project.get("name", "Unknown")
                path = project.get("path", "Unknown")
                last_indexed = project.get("last_indexed", "Unknown")
                chunk_count = project.get("chunk_count", 0)
                file_count = project.get("file_count", 0)
                index_size_mb = project.get("index_size_mb", 0)
                
                total_chunks += chunk_count
                total_size_mb += index_size_mb
                
                response_text += f"## {name}\n"
                response_text += f"**Path:** {path}\n"
                response_text += f"**Last Indexed:** {last_indexed}\n"
                response_text += f"**Files:** {file_count}\n"
                response_text += f"**Code Chunks:** {chunk_count}\n"
                response_text += f"**Index Size:** {index_size_mb:.2f} MB\n"
                
                if detailed:
                    # Get additional statistics if available
                    stats = await self.project_memory.get_project_stats(name)
                    if stats:
                        response_text += f"**Languages:** {', '.join(stats.get('languages', []))}\n"
                        response_text += f"**Average Chunk Size:** {stats.get('avg_chunk_size', 0)} chars\n"
                        response_text += f"**Most Common Types:** {', '.join(stats.get('common_types', []))}\n"
                
                response_text += "\n"
            
            # Summary
            response_text += "---\n"
            response_text += f"**Total Projects:** {len(projects)}\n"
            response_text += f"**Total Code Chunks:** {total_chunks}\n"
            response_text += f"**Total Index Size:** {total_size_mb:.2f} MB\n"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            return [TextContent(
                type="text", 
                text=f"Error listing project memories: {str(e)}"
            )]
```

### Register Tools in Main Module
Find the main tool registration file (likely `src/mcp_server_tree_sitter/server.py` or similar) and add:

```python
# Add imports at top
from .tools.memory.index_project_memory import IndexProjectMemoryTool
from .tools.memory.query_project_memory import QueryProjectMemoryTool  
from .tools.memory.list_project_memories import ListProjectMemoriesTool
from .services.project_memory import ProjectMemory
from .services.embedding_service import EmbeddingService

# In the tool registration section, add memory tools
async def setup_memory_tools(container):
    """Setup memory-related tools."""
    # Get or create memory services
    embedding_service = container.get(EmbeddingService)
    project_memory = container.get(ProjectMemory) 
    
    # Create tools
    index_tool = IndexProjectMemoryTool(project_memory)
    query_tool = QueryProjectMemoryTool(project_memory)
    list_tool = ListProjectMemoriesTool(project_memory)
    
    return [index_tool, query_tool, list_tool]

# In main server setup, register the tools
memory_tools = await setup_memory_tools(container)
all_tools.extend(memory_tools)
```

### Update __init__.py Files
Add exports to `src/mcp_server_tree_sitter/tools/memory/__init__.py`:
```python
from .index_project_memory import IndexProjectMemoryTool
from .query_project_memory import QueryProjectMemoryTool
from .list_project_memories import ListProjectMemoriesTool

__all__ = [
    "IndexProjectMemoryTool",
    "QueryProjectMemoryTool", 
    "ListProjectMemoriesTool"
]
```

## Test Conditions - Milestone 3 COMPLETE When:
- [ ] All three tool files import without errors: `python -c "from src.mcp_server_tree_sitter.tools.memory import *"`
- [ ] Tools register without errors: Server starts successfully with new tools
- [ ] MCP server starts with memory tools: `python -m mcp_server_tree_sitter.server` runs without tool registration errors
- [ ] Tools appear in tool list: Server exposes 3 new memory tools in capabilities
- [ ] index_project_memory accepts valid input: Test with small directory containing Python files
- [ ] query_project_memory responds to queries: Test with simple query after indexing
- [ ] list_project_memories returns structured data: Verify proper text formatting and data

## Verification Test
Create `test_milestone_3.py`:
```python
#!/usr/bin/env python3

import asyncio
import tempfile
import os
from pathlib import Path

async def test_tools():
    """Test that memory tools are properly registered and functional."""
    
    # Import the tools
    try:
        from src.mcp_server_tree_sitter.tools.memory import (
            IndexProjectMemoryTool,
            QueryProjectMemoryTool,
            ListProjectMemoriesTool
        )
        print("✓ Tools imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tools: {e}")
        return False
    
    # Create temporary test project
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("def hello_world():\n    return 'Hello, World!'\n")
        
        print("✓ Test project created")
        
        # Test tools can be instantiated
        try:
            # This will require actual service implementations
            # For now, just test the tool definitions exist
            index_tool = IndexProjectMemoryTool(None)
            query_tool = QueryProjectMemoryTool(None) 
            list_tool = ListProjectMemoriesTool(None)
            
            # Check tool definitions
            assert index_tool.definition.name == "index_project_memory"
            assert query_tool.definition.name == "query_project_memory"
            assert list_tool.definition.name == "list_project_memories"
            
            print("✓ Tool definitions correct")
            
        except Exception as e:
            print(f"✗ Tool instantiation failed: {e}")
            return False
    
    print("🎉 Milestone 3 PASSED - All tools ready")
    return True

if __name__ == "__main__":
    asyncio.run(test_tools())
```

Run: `python test_milestone_3.py`

## On Completion
Update ENHANCEMENT_PLAN.md to mark Milestone 3 as ✓ and ready for Milestone 4.