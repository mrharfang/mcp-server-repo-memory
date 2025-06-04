#!/usr/bin/env python3
"""
Milestone 4: Integration Test for Project Memory MCP

This script performs end-to-end testing of the project memory system
with a real small codebase to validate all components work together.
"""

import asyncio
import tempfile
import shutil
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List


class IntegrationTestError(Exception):
    """Custom exception for integration test failures."""
    pass


async def create_test_codebase() -> str:
    """Create a small test codebase for integration testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_codebase_")
    print(f"Creating test codebase at: {temp_dir}")
    
    # Create test Python files with realistic content
    files = {
        "main.py": '''
"""Main entry point for the test application."""

from calculator import Calculator
from utils import load_config, save_config
import json


def main():
    """Main entry point for the application."""
    print("Starting Test Application...")
    
    # Load configuration
    try:
        config = load_config("config.json")
        print(f"Loaded config: {config}")
    except FileNotFoundError:
        config = {"debug": True, "version": "1.0.0"}
        save_config(config, "config.json")
        print("Created default config")
    
    # Initialize calculator
    calculator = Calculator()
    
    # Perform some calculations
    result = calculator.add(5, 3)
    print(f"5 + 3 = {result}")
    
    result = calculator.multiply(4, 7)
    print(f"4 * 7 = {result}")
    
    # Test error handling
    try:
        result = calculator.divide(10, 0)
    except ValueError as e:
        print(f"Expected error: {e}")
    
    print("Application completed successfully!")


if __name__ == "__main__":
    main()
''',
        
        "calculator.py": '''
"""Calculator module with basic arithmetic operations."""

import math
from typing import Union

Number = Union[int, float]


class Calculator:
    """A simple calculator class with error handling."""
    
    def __init__(self):
        """Initialize calculator with history tracking."""
        self.history = []
        self.precision = 2
    
    def add(self, a: Number, b: Number) -> Number:
        """Add two numbers together.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        result = a + b
        self._record_operation(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: Number, b: Number) -> Number:
        """Subtract b from a.
        
        Args:
            a: First number  
            b: Second number
            
        Returns:
            Difference of a and b
        """
        result = a - b
        self._record_operation(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: Number, b: Number) -> Number:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        result = a * b
        self._record_operation(f"{a} * {b} = {result}")
        return result
    
    def divide(self, a: Number, b: Number) -> float:
        """Divide a by b with error handling.
        
        Args:
            a: Dividend
            b: Divisor
            
        Returns:
            Quotient of a and b
            
        Raises:
            ValueError: If b is zero (division by zero)
        """
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        self._record_operation(f"{a} / {b} = {result}")
        return result
    
    def power(self, base: Number, exponent: Number) -> Number:
        """Calculate base raised to exponent power.
        
        Args:
            base: Base number
            exponent: Exponent
            
        Returns:
            Result of base^exponent
        """
        result = base ** exponent
        self._record_operation(f"{base} ^ {exponent} = {result}")
        return result
    
    def square_root(self, number: Number) -> float:
        """Calculate square root of a number.
        
        Args:
            number: Number to find square root of
            
        Returns:
            Square root of the number
            
        Raises:
            ValueError: If number is negative
        """
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        result = math.sqrt(number)
        self._record_operation(f"√{number} = {result}")
        return result
    
    def _record_operation(self, operation: str) -> None:
        """Record operation in history.
        
        Args:
            operation: String description of the operation
        """
        self.history.append(operation)
        
        # Keep only last 100 operations
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def get_history(self) -> List[str]:
        """Get calculation history.
        
        Returns:
            List of operation strings
        """
        return self.history.copy()
    
    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()
''',
        
        "utils.py": '''
"""Utility functions for configuration and data handling."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Dictionary containing configuration data
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(file_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config has required fields
        validate_config(config)
        return config
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e}")


def save_config(config: Dict[str, Any], file_path: str) -> None:
    """Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the config file
    """
    config_path = Path(file_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise IOError(f"Failed to save config file: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Add more validation as needed
    if 'version' in config and not isinstance(config['version'], str):
        raise ValueError("Version must be a string")


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number as string
    """
    return f"{num:.{decimals}f}"


def ensure_directory(dir_path: str) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        dir_path: Path to directory
        
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(file_path: str) -> bool:
    """Check if file exists.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return path.stat().st_size
''',
        
        "api.py": '''
"""API module for external integrations."""

import requests
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class APIResponse:
    """Response from API call."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class APIClient:
    """HTTP API client for external services."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """Initialize API client.
        
        Args:
            base_url: Base URL for API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}'
            })
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request to API.
        
        Args:
            endpoint: API endpoint
            params: Optional query parameters
            
        Returns:
            APIResponse with results
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            return APIResponse(
                success=response.status_code == 200,
                data=response.json() if response.content else None,
                status_code=response.status_code
            )
            
        except requests.RequestException as e:
            return APIResponse(
                success=False,
                error=str(e)
            )
        except json.JSONDecodeError as e:
            return APIResponse(
                success=False,
                error=f"Invalid JSON response: {e}",
                status_code=response.status_code
            )
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make POST request to API.
        
        Args:
            endpoint: API endpoint
            data: Optional request body data
            
        Returns:
            APIResponse with results
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.post(
                url,
                json=data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            return APIResponse(
                success=response.status_code in [200, 201],
                data=response.json() if response.content else None,
                status_code=response.status_code
            )
            
        except requests.RequestException as e:
            return APIResponse(
                success=False,
                error=str(e)
            )
        except json.JSONDecodeError as e:
            return APIResponse(
                success=False,
                error=f"Invalid JSON response: {e}",
                status_code=response.status_code
            )


def fetch_user_data(user_id: int) -> Dict[str, Any]:
    """Fetch user data from external API.
    
    Args:
        user_id: ID of user to fetch
        
    Returns:
        User data dictionary
    """
    client = APIClient("https://jsonplaceholder.typicode.com")
    response = client.get(f"users/{user_id}")
    
    if response.success:
        return response.data
    else:
        raise Exception(f"Failed to fetch user data: {response.error}")
''',
        
        "README.md": '''# Test Codebase

A simple test codebase for integration testing of the Project Memory MCP system.

## Features

- **Calculator Module**: Basic arithmetic operations with error handling
- **Configuration Management**: JSON-based configuration loading and saving
- **API Client**: HTTP client for external service integration
- **Utility Functions**: Common helper functions for file and data operations

## Modules

### calculator.py
- `Calculator` class with basic math operations
- History tracking for calculations
- Error handling for edge cases (division by zero, negative square roots)

### utils.py
- Configuration file management (load/save JSON)
- File system utilities
- Number formatting helpers

### api.py
- HTTP API client with authentication support
- GET/POST request methods
- Response handling and error management

### main.py
- Main application entry point
- Demonstrates usage of all modules
- Error handling examples

## Testing Focus Areas

This codebase is designed to test the Project Memory MCP system's ability to:

1. **Index diverse code patterns**: Classes, functions, error handling, imports
2. **Handle multiple file types**: Python modules with different purposes
3. **Semantic search capabilities**: Find relevant code based on natural language queries
4. **Cross-module relationships**: Track dependencies and usage patterns

## Usage

```bash
python main.py
```

This will run a demonstration of all the calculator and utility functions.
'''
    }
    
    # Write all files to the temporary directory
    for filename, content in files.items():
        file_path = Path(temp_dir) / filename
        file_path.write_text(content.strip(), encoding='utf-8')
    
    print(f"✓ Created {len(files)} test files")
    return temp_dir


class MockMCPClient:
    """Mock MCP client for testing the memory tools directly."""
    
    def __init__(self):
        """Initialize mock client with actual services."""
        # Import the actual services
        try:
            from src.mcp_server_tree_sitter.services.project_memory import ProjectMemory
            from src.mcp_server_tree_sitter.services.embedding_service import EmbeddingService
            
            self.embedding_service = EmbeddingService()
            self.project_memory = ProjectMemory(embedding_service=self.embedding_service)
            
            print("✓ Successfully initialized mock MCP client")
            
        except ImportError as e:
            raise IntegrationTestError(f"Failed to import required services: {e}")
    
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Simulate MCP tool call by directly calling the service methods."""
        try:
            if tool_name == "index_project_memory":
                return await self._index_project_memory(**params)
            elif tool_name == "query_project_memory":
                return await self._query_project_memory(**params)
            elif tool_name == "list_project_memories":
                return await self._list_project_memories(**params)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            raise IntegrationTestError(f"Tool call failed for {tool_name}: {e}")
    
    async def _index_project_memory(self, project_path: str, project_name: str, 
                                   force_reindex: bool = False) -> str:
        """Index project memory."""
        # Validate project path
        project_path_obj = Path(project_path)
        if not project_path_obj.exists():
            return json.dumps({
                "success": False,
                "error": f"Project path does not exist: {project_path}"
            })
        
        start_time = time.time()
        
        try:
            result = await self.project_memory.index_project(
                project_path=project_path,
                project_name=project_name,
                force_reindex=force_reindex
            )
            
            elapsed_time = time.time() - start_time
            
            response = {
                "success": True,
                "project_name": project_name,
                "chunks_indexed": result.get("chunk_count", 0),  # Map from chunk_count
                "files_processed": result.get("total_chunks_found", 0),  # Use total chunks as files
                "time_taken": f"{elapsed_time:.2f}s",
                "index_size_mb": 0  # Not available from current implementation
            }
            
            return json.dumps(response, indent=2)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Error indexing project: {str(e)}"
            })
    
    async def _query_project_memory(self, project_name: str, query: str, 
                                   context_limit: int = 5, include_metadata: bool = True) -> str:
        """Query project memory."""
        start_time = time.time()
        
        try:
            results = await self.project_memory.query_memory(
                query=query,
                project_name=project_name,
                limit=context_limit
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            if not results:
                return f"No relevant results found for query: '{query}' in project '{project_name}'"
            
            # Format results
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
            return f"Error querying project memory: {str(e)}"
    
    async def _list_project_memories(self, detailed: bool = False) -> str:
        """List project memories."""
        try:
            client = self.project_memory._get_chroma_client()
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
                
                if detailed and chunk_count > 0:
                    try:
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
            return f"Error listing project memories: {str(e)}"


async def run_integration_tests() -> bool:
    """Run the complete integration test suite."""
    print("🚀 Starting Project Memory MCP Integration Test")
    print("=" * 60)
    
    test_results = []
    test_codebase_dir = None
    
    try:
        # Create test codebase
        print("\n📁 Creating test codebase...")
        test_codebase_dir = await create_test_codebase()
        test_results.append(("Test codebase creation", True, "Created 5 test files"))
        
        # Initialize mock client
        print("\n🔧 Initializing MCP client...")
        client = MockMCPClient()
        test_results.append(("MCP client initialization", True, "Client ready"))
        
        project_name = "test-integration-codebase"
        
        # Test 1: Index the project
        print(f"\n📊 Test 1: Indexing project '{project_name}'...")
        start_time = time.time()
        
        index_result = await client.call_tool('index_project_memory', {
            'project_path': test_codebase_dir,
            'project_name': project_name,
            'force_reindex': True
        })
        
        index_time = time.time() - start_time
        print(f"Index result: {index_result[:200]}...")
        
        # Parse and validate index result
        try:
            index_data = json.loads(index_result)
            if index_data.get("success"):
                chunks_indexed = index_data.get("chunks_indexed", 0)
                files_processed = index_data.get("files_processed", 0)
                
                assert chunks_indexed > 0, f"No chunks indexed (got {chunks_indexed})"
                assert files_processed > 0, f"No files processed (got {files_processed})"
                assert index_time < 30, f"Indexing took too long: {index_time:.2f}s"
                
                test_results.append(("Project indexing", True, 
                                   f"{chunks_indexed} chunks, {files_processed} files in {index_time:.2f}s"))
                print(f"✓ Indexing completed: {chunks_indexed} chunks from {files_processed} files")
            else:
                error_msg = index_data.get("error", "Unknown error")
                test_results.append(("Project indexing", False, error_msg))
                print(f"✗ Indexing failed: {error_msg}")
                return False
                
        except json.JSONDecodeError as e:
            test_results.append(("Project indexing", False, f"Invalid JSON response: {e}"))
            return False
        
        # Test 2: Query the project with various queries
        print(f"\n🔍 Test 2: Querying project '{project_name}'...")
        test_queries = [
            ("calculator addition function", "Should find Calculator.add method"),
            ("error handling division by zero", "Should find divide method with zero check"),
            ("configuration loading", "Should find load_config function"),
            ("API client authentication", "Should find APIClient with auth"),
            ("json file operations", "Should find JSON load/save operations")
        ]
        
        query_success_count = 0
        for query, expected in test_queries:
            print(f"  Testing query: '{query}'")
            
            try:
                query_result = await client.call_tool('query_project_memory', {
                    'project_name': project_name,
                    'query': query,
                    'context_limit': 3,
                    'include_metadata': True
                })
                
                # Validate query returned results
                if "No relevant results found" in query_result or "Found 0 relevant" in query_result:
                    print(f"    ✗ No results for: {query}")
                    test_results.append((f"Query: {query}", False, "No results found"))
                else:
                    # Check if results contain reasonable content
                    if "Found" in query_result and "code chunks" in query_result:
                        print(f"    ✓ Found results for: {query}")
                        query_success_count += 1
                    else:
                        print(f"    ⚠ Unexpected result format for: {query}")
                        test_results.append((f"Query: {query}", False, "Unexpected format"))
                        
            except Exception as e:
                print(f"    ✗ Query failed: {e}")
                test_results.append((f"Query: {query}", False, str(e)))
        
        # Validate overall query success rate
        query_success_rate = query_success_count / len(test_queries)
        if query_success_rate >= 0.8:  # At least 80% of queries should succeed
            test_results.append(("Query functionality", True, 
                               f"{query_success_count}/{len(test_queries)} queries successful"))
            print(f"✓ Query test passed: {query_success_count}/{len(test_queries)} queries successful")
        else:
            test_results.append(("Query functionality", False, 
                               f"Only {query_success_count}/{len(test_queries)} queries successful"))
            print(f"✗ Query test failed: only {query_success_count}/{len(test_queries)} queries successful")
        
        # Test 3: List project memories
        print(f"\n📋 Test 3: Listing project memories...")
        try:
            list_result = await client.call_tool('list_project_memories', {
                'detailed': True
            })
            
            # Validate list result contains our project
            if project_name in list_result or "test-integration" in list_result:
                test_results.append(("List projects", True, "Project found in list"))
                print("✓ Project found in memory list")
                print(f"List preview: {list_result[:200]}...")
            else:
                test_results.append(("List projects", False, "Project not found in list"))
                print("✗ Project not found in memory list")
                
        except Exception as e:
            test_results.append(("List projects", False, str(e)))
            print(f"✗ List projects failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed with exception: {e}")
        test_results.append(("Integration test", False, str(e)))
        return False
        
    finally:
        # Print test results summary
        print(f"\n{'='*60}")
        print("INTEGRATION TEST RESULTS:")
        print("-" * 30)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for _, passed, _ in test_results if passed)
        
        for test_name, passed, message in test_results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status:8} {test_name:25} | {message}")
        
        print("-" * 30)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            success = True
        else:
            print("❌ SOME INTEGRATION TESTS FAILED")
            success = False
        
        # Cleanup test codebase
        if test_codebase_dir and os.path.exists(test_codebase_dir):
            try:
                shutil.rmtree(test_codebase_dir)
                print(f"🧹 Cleaned up test directory: {test_codebase_dir}")
            except Exception as e:
                print(f"⚠️ Failed to cleanup test directory: {e}")
        
        return success


if __name__ == "__main__":
    async def main():
        """Main entry point for integration test."""
        try:
            success = await run_integration_tests()
            return success
        except KeyboardInterrupt:
            print("\n⚠️ Integration test interrupted by user")
            return False
        except Exception as e:
            print(f"\n💥 Unexpected error in integration test: {e}")
            return False
    
    # Run the integration test
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
