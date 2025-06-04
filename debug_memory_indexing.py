#!/usr/bin/env python3
"""Debug script to test memory indexing and identify issues."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_server_tree_sitter.services.project_memory import ProjectMemory
from mcp_server_tree_sitter.services.embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def debug_indexing():
    """Debug the indexing process."""
    print("🔍 Starting memory indexing debug...")
    
    # Get current project path
    project_path = Path(__file__).parent.absolute()
    project_name = "mcp-server-repo-memory"
    
    print(f"📁 Project path: {project_path}")
    print(f"📝 Project name: {project_name}")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        project_memory = ProjectMemory(embedding_service)
        
        print("✅ Services initialized")
        
        # Test indexing
        result = await project_memory.index_project(
            project_path=str(project_path),
            project_name=project_name,
            force_reindex=True
        )
        
        print(f"📊 Indexing result: {result}")
        
        # Print detailed results
        if result.get('status') == 'indexed':
            print(f"✅ Success!")
            print(f"   • Chunks indexed: {result.get('chunks_indexed', 0)}")
            print(f"   • Files processed: {result.get('files_processed', 0)}")
            print(f"   • Total chunks found: {result.get('total_chunks_found', 0)}")
        else:
            print(f"❌ Failed or no chunks indexed")
            print(f"   • Status: {result.get('status')}")
            print(f"   • Chunks: {result.get('chunk_count', 0)}")
            
    except Exception as e:
        print(f"💥 Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_indexing())
