#!/usr/bin/env python3
"""Functional test for memory tools with actual data."""

import asyncio
import tempfile
import json
from pathlib import Path
from src.mcp_server_tree_sitter.services.embedding_service import EmbeddingService
from src.mcp_server_tree_sitter.services.project_memory import ProjectMemory


async def test_memory_tools_functional():
    """Test memory tools with real data flow."""
    print("🧪 Testing Memory Tools Functional Flow")
    print("=" * 50)
    
    # Set up services
    embedding_service = EmbeddingService()
    project_memory = ProjectMemory(embedding_service)
    
    # Create a test project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create some test Python files
        (project_path / "main.py").write_text("""
def hello_world():
    '''Greets the world.'''
    print("Hello, World!")

def calculate_sum(a, b):
    '''Calculates the sum of two numbers.'''
    return a + b

if __name__ == "__main__":
    hello_world()
    result = calculate_sum(5, 3)
    print(f"Sum: {result}")
""")
        
        (project_path / "utils.py").write_text("""
def format_name(first_name, last_name):
    '''Formats a full name from first and last name.'''
    return f"{first_name} {last_name}"

def validate_email(email):
    '''Basic email validation.'''
    return "@" in email and "." in email
""")
        
        project_name = "test_project"
        
        # Test 1: Index the project
        print("1. Testing index_project_memory...")
        try:
            result = await project_memory.index_project(
                project_path=str(project_path),
                project_name=project_name,
                force_reindex=True
            )
            print(f"   ✓ Indexing result: {result}")
            
            if result.get('chunk_count', 0) > 0:
                print(f"   ✓ Successfully indexed {result['chunk_count']} chunks")
            else:
                print("   ⚠ Warning: No chunks were indexed")
                
        except Exception as e:
            print(f"   ✗ Error indexing: {e}")
            return False
        
        # Test 2: Query the memory
        print("\n2. Testing query_project_memory...")
        try:
            results = await project_memory.query_memory(
                query="function that calculates sum",
                project_name=project_name,
                limit=3
            )
            print(f"   ✓ Query returned {len(results)} results")
            
            if results:
                for i, result in enumerate(results):
                    print(f"   • Result {i+1}: {result.chunk.content[:50]}... (similarity: {result.similarity:.3f})")
            else:
                print("   ⚠ Warning: No results found")
                
        except Exception as e:
            print(f"   ✗ Error querying: {e}")
            return False
        
        # Test 3: List projects
        print("\n3. Testing list_project_memories...")
        try:
            client = project_memory._get_chroma_client()
            collections = client.list_collections()
            print(f"   ✓ Found {len(collections)} collections")
            
            for collection in collections:
                count = collection.count()
                print(f"   • Collection: {collection.name} ({count} chunks)")
                
        except Exception as e:
            print(f"   ✗ Error listing: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("🎉 All functional tests passed!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_memory_tools_functional())
    exit(0 if success else 1)
