#!/usr/bin/env python3
"""Test script to verify Milestone 3 MCP tools implementation.

This test validates that the three new memory tools are properly registered
and can be imported without errors.
"""

import asyncio
import tempfile
import os
import sys
from pathlib import Path


def test_memory_tools_import():
    """Test that memory tools can be imported successfully."""
    try:
        from src.mcp_server_tree_sitter.tools.memory_tools import register_memory_tools
        print("✓ Memory tools imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import memory tools: {e}")
        return False


def test_registration_import():
    """Test that the updated registration module imports successfully."""
    try:
        from src.mcp_server_tree_sitter.tools.registration import register_tools
        print("✓ Registration module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import registration module: {e}")
        return False


def test_memory_services_import():
    """Test that memory services can be imported."""
    try:
        from src.mcp_server_tree_sitter.services.project_memory import ProjectMemory
        from src.mcp_server_tree_sitter.services.embedding_service import EmbeddingService
        print("✓ Memory services imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import memory services: {e}")
        return False


def test_memory_tool_definitions():
    """Test that memory tools have correct definitions."""
    try:
        from src.mcp_server_tree_sitter.tools.memory_tools import register_memory_tools
        from src.mcp_server_tree_sitter.services.project_memory import ProjectMemory
        from src.mcp_server_tree_sitter.services.embedding_service import EmbeddingService
        
        # Create mock services
        class MockMCPServer:
            def __init__(self):
                self.tools = []
            
            def tool(self):
                def decorator(func):
                    self.tools.append(func.__name__)
                    return func
                return decorator
        
        # Create mock memory services
        embedding_service = EmbeddingService()
        project_memory = ProjectMemory(embedding_service)
        mock_server = MockMCPServer()
        
        # Register tools
        register_memory_tools(mock_server, project_memory, embedding_service)
        
        # Check that tools were registered
        expected_tools = ["index_project_memory", "query_project_memory", "list_project_memories"]
        for tool_name in expected_tools:
            if tool_name in mock_server.tools:
                print(f"✓ Tool '{tool_name}' registered successfully")
            else:
                print(f"✗ Tool '{tool_name}' not found in registered tools")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Tool definition test failed: {e}")
        return False


def test_project_structure():
    """Test that the project structure is correct."""
    try:
        # Check that memory_tools.py exists
        memory_tools_path = Path("src/mcp_server_tree_sitter/tools/memory_tools.py")
        if memory_tools_path.exists():
            print("✓ memory_tools.py file exists")
        else:
            print("✗ memory_tools.py file not found")
            return False
        
        # Check that __init__.py was updated
        init_path = Path("src/mcp_server_tree_sitter/tools/__init__.py")
        if init_path.exists():
            content = init_path.read_text()
            if "register_memory_tools" in content:
                print("✓ __init__.py updated with memory tools")
            else:
                print("✗ __init__.py not updated with memory tools")
                return False
        else:
            print("✗ __init__.py file not found")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Project structure test failed: {e}")
        return False


def main():
    """Run all tests for Milestone 3."""
    print("🧪 Testing Milestone 3: MCP Tool Extensions")
    print("=" * 50)
    
    tests = [
        test_memory_tools_import,
        test_registration_import, 
        test_memory_services_import,
        test_project_structure,
        test_memory_tool_definitions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Milestone 3 PASSED - All memory tools ready!")
        print("\nMemory tools successfully implemented:")
        print("  • index_project_memory - Create/update semantic memory index")
        print("  • query_project_memory - Semantic search with natural language")
        print("  • list_project_memories - Show indexed projects and statistics")
        return True
    else:
        print("❌ Milestone 3 FAILED - Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
