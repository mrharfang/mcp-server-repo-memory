"""
Comprehensive test suite for Project Memory MCP - Milestone 4
Tests edge cases, error conditions, and real-world scenarios.
"""

import asyncio
import tempfile
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from integration_test import MockMCPClient, IntegrationTestError


class EdgeCaseTests:
    """Test suite for edge cases and error conditions."""
    
    def __init__(self):
        """Initialize edge case test suite."""
        self.client = None
        self.test_results = []
    
    async def setup(self):
        """Set up the test environment."""
        self.client = MockMCPClient()
        print("✓ Edge case test setup complete")
    
    async def test_empty_project(self) -> Dict[str, Any]:
        """Test indexing an empty project directory."""
        print("Testing empty project indexing...")
        
        # Create empty directory
        temp_dir = tempfile.mkdtemp(prefix="empty_project_")
        
        try:
            result = await self.client.call_tool('index_project_memory', {
                'project_path': temp_dir,
                'project_name': 'empty-project',
                'force_reindex': True
            })
            
            result_data = json.loads(result)
            
            if result_data.get("success") and result_data.get("chunks_indexed", 0) == 0:
                return {"test": "empty_project", "status": "PASS", "message": "Handled empty project correctly"}
            else:
                return {"test": "empty_project", "status": "FAIL", "message": f"Unexpected result: {result}"}
                
        except Exception as e:
            return {"test": "empty_project", "status": "ERROR", "message": str(e)}
        finally:
            os.rmdir(temp_dir)
    
    async def test_nonexistent_project(self) -> Dict[str, Any]:
        """Test indexing a non-existent project path."""
        print("Testing non-existent project path...")
        
        fake_path = "/this/path/does/not/exist"
        
        try:
            result = await self.client.call_tool('index_project_memory', {
                'project_path': fake_path,
                'project_name': 'nonexistent-project',
                'force_reindex': True
            })
            
            result_data = json.loads(result)
            
            if not result_data.get("success") and "does not exist" in result_data.get("error", ""):
                return {"test": "nonexistent_project", "status": "PASS", "message": "Correctly handled non-existent path"}
            else:
                return {"test": "nonexistent_project", "status": "FAIL", "message": f"Did not handle error correctly: {result}"}
                
        except Exception as e:
            return {"test": "nonexistent_project", "status": "ERROR", "message": str(e)}
    
    async def test_corrupted_files(self) -> Dict[str, Any]:
        """Test handling files with unusual content or encoding issues."""
        print("Testing corrupted/unusual files...")
        
        temp_dir = tempfile.mkdtemp(prefix="corrupted_project_")
        
        try:
            # Create files with different issues
            files = {
                "binary_file.pyc": b"\x00\x01\x02\x03\x04\x05",  # Binary content
                "empty_file.py": "",  # Empty file
                "very_long_line.py": "x = " + "a" * 10000 + "\n",  # Very long line
                "mixed_encoding.py": "# -*- coding: utf-8 -*-\nprint('Hello 世界')\n",  # Unicode
                "syntax_error.py": "def broken_function(\n    pass\n",  # Syntax error
                "large_file.py": "# Large file\n" + ("print('line')\n" * 1000),  # Large file
            }
            
            for filename, content in files.items():
                file_path = Path(temp_dir) / filename
                if isinstance(content, bytes):
                    file_path.write_bytes(content)
                else:
                    file_path.write_text(content, encoding='utf-8')
            
            # Try to index
            result = await self.client.call_tool('index_project_memory', {
                'project_path': temp_dir,
                'project_name': 'corrupted-project',
                'force_reindex': True
            })
            
            result_data = json.loads(result)
            
            # Should succeed but may have fewer chunks due to filtering
            if result_data.get("success"):
                chunks = result_data.get("chunks_indexed", 0)
                return {"test": "corrupted_files", "status": "PASS", 
                       "message": f"Handled corrupted files, indexed {chunks} chunks"}
            else:
                error = result_data.get("error", "Unknown error")
                return {"test": "corrupted_files", "status": "FAIL", "message": f"Failed to handle corrupted files: {error}"}
                
        except Exception as e:
            return {"test": "corrupted_files", "status": "ERROR", "message": str(e)}
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_query_edge_cases(self) -> Dict[str, Any]:
        """Test querying with edge case queries."""
        print("Testing query edge cases...")
        
        # First create a simple project to query
        temp_dir = tempfile.mkdtemp(prefix="query_test_")
        
        try:
            # Create a simple test file
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("""
def hello_world():
    print("Hello, World!")
    return True

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""")
            
            # Index the project
            index_result = await self.client.call_tool('index_project_memory', {
                'project_path': temp_dir,
                'project_name': 'query-edge-test',
                'force_reindex': True
            })
            
            index_data = json.loads(index_result)
            if not index_data.get("success"):
                return {"test": "query_edge_cases", "status": "FAIL", "message": "Failed to index test project"}
            
            # Test various edge case queries
            edge_queries = [
                "",  # Empty query
                " ",  # Whitespace only
                "a",  # Single character
                "nonexistent_function_that_should_not_be_found",  # No matches expected
                "hello world test class value 42",  # Multiple terms
                "特殊字符",  # Non-ASCII characters
                "SELECT * FROM table",  # SQL-like query
                "function(arg1, arg2)",  # Code-like query
                "HELLO WORLD",  # All caps
                "hello world" * 50,  # Very long query
            ]
            
            edge_results = []
            
            for query in edge_queries:
                try:
                    result = await self.client.call_tool('query_project_memory', {
                        'project_name': 'query-edge-test',
                        'query': query,
                        'context_limit': 3,
                        'include_metadata': True
                    })
                    
                    # Should not crash, even if no results
                    if "No relevant results found" in result or "Found" in result:
                        edge_results.append((query, "OK"))
                    else:
                        edge_results.append((query, "UNEXPECTED"))
                        
                except Exception as e:
                    edge_results.append((query, f"ERROR: {e}"))
            
            # Check results
            ok_count = sum(1 for _, status in edge_results if status == "OK")
            total_count = len(edge_results)
            
            if ok_count == total_count:
                return {"test": "query_edge_cases", "status": "PASS", 
                       "message": f"All {total_count} edge case queries handled correctly"}
            else:
                failed_queries = [(q, s) for q, s in edge_results if s != "OK"]
                return {"test": "query_edge_cases", "status": "PARTIAL", 
                       "message": f"{ok_count}/{total_count} queries OK. Failed: {failed_queries[:3]}"}
                
        except Exception as e:
            return {"test": "query_edge_cases", "status": "ERROR", "message": str(e)}
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent indexing and querying operations."""
        print("Testing concurrent operations...")
        
        try:
            # Create multiple small projects
            projects = []
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f"concurrent_test_{i}_")
                test_file = Path(temp_dir) / f"module_{i}.py"
                test_file.write_text(f"""
def function_{i}():
    return {i}

class Class_{i}:
    value = {i}
""")
                projects.append((temp_dir, f"concurrent-test-{i}"))
            
            # Index all projects concurrently
            index_tasks = []
            for project_path, project_name in projects:
                task = self.client.call_tool('index_project_memory', {
                    'project_path': project_path,
                    'project_name': project_name,
                    'force_reindex': True
                })
                index_tasks.append(task)
            
            # Wait for all indexing to complete
            index_results = await asyncio.gather(*index_tasks, return_exceptions=True)
            
            # Check if all indexing succeeded
            success_count = 0
            for result in index_results:
                if not isinstance(result, Exception):
                    try:
                        data = json.loads(result)
                        if data.get("success"):
                            success_count += 1
                    except json.JSONDecodeError:
                        pass
            
            if success_count == len(projects):
                return {"test": "concurrent_operations", "status": "PASS", 
                       "message": f"All {len(projects)} concurrent indexing operations succeeded"}
            else:
                return {"test": "concurrent_operations", "status": "FAIL", 
                       "message": f"Only {success_count}/{len(projects)} concurrent operations succeeded"}
                
        except Exception as e:
            return {"test": "concurrent_operations", "status": "ERROR", "message": str(e)}
        finally:
            # Cleanup
            import shutil
            for project_path, _ in projects:
                shutil.rmtree(project_path, ignore_errors=True)
    
    async def test_memory_persistence(self) -> Dict[str, Any]:
        """Test that memory persists across client instances."""
        print("Testing memory persistence...")
        
        temp_dir = tempfile.mkdtemp(prefix="persistence_test_")
        
        try:
            # Create a test project
            test_file = Path(temp_dir) / "persistent.py"
            test_file.write_text("""
def persistent_function():
    return "I should be remembered"

class PersistentClass:
    def remember_me(self):
        return "persistence test"
""")
            
            # Index with first client instance
            project_name = "persistence-test"
            result1 = await self.client.call_tool('index_project_memory', {
                'project_path': temp_dir,
                'project_name': project_name,
                'force_reindex': True
            })
            
            data1 = json.loads(result1)
            if not data1.get("success"):
                return {"test": "memory_persistence", "status": "FAIL", "message": "Initial indexing failed"}
            
            # Create a new client instance
            new_client = MockMCPClient()
            
            # Query with the new client
            result2 = await new_client.call_tool('query_project_memory', {
                'project_name': project_name,
                'query': 'persistent function',
                'context_limit': 3,
                'include_metadata': True
            })
            
            if "persistent_function" in result2 or "Found" in result2:
                return {"test": "memory_persistence", "status": "PASS", 
                       "message": "Memory persisted across client instances"}
            else:
                return {"test": "memory_persistence", "status": "FAIL", 
                       "message": "Memory did not persist across client instances"}
                
        except Exception as e:
            return {"test": "memory_persistence", "status": "ERROR", "message": str(e)}
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_large_query_results(self) -> Dict[str, Any]:
        """Test handling of queries that return many results."""
        print("Testing large query results...")
        
        temp_dir = tempfile.mkdtemp(prefix="large_results_test_")
        
        try:
            # Create a file with many similar functions
            test_file = Path(temp_dir) / "many_functions.py"
            functions_code = "\n".join([
                f"def test_function_{i}():\n    return {i}\n" 
                for i in range(20)
            ])
            test_file.write_text(functions_code)
            
            # Index the project
            result = await self.client.call_tool('index_project_memory', {
                'project_path': temp_dir,
                'project_name': 'large-results-test',
                'force_reindex': True
            })
            
            data = json.loads(result)
            if not data.get("success"):
                return {"test": "large_query_results", "status": "FAIL", "message": "Indexing failed"}
            
            # Query for something that should match many functions
            query_result = await self.client.call_tool('query_project_memory', {
                'project_name': 'large-results-test',
                'query': 'test function',
                'context_limit': 10,  # Request many results
                'include_metadata': True
            })
            
            # Should handle large results gracefully
            if "Found" in query_result and len(query_result) > 1000:  # Should be substantial
                return {"test": "large_query_results", "status": "PASS", 
                       "message": "Handled large query results correctly"}
            else:
                return {"test": "large_query_results", "status": "PARTIAL", 
                       "message": f"Query result size: {len(query_result)} chars"}
                
        except Exception as e:
            return {"test": "large_query_results", "status": "ERROR", "message": str(e)}
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def run_all_edge_case_tests(self) -> Dict[str, Any]:
        """Run all edge case tests."""
        print("🧪 Starting Edge Case Test Suite")
        print("=" * 40)
        
        await self.setup()
        
        # Define all tests
        test_methods = [
            self.test_empty_project,
            self.test_nonexistent_project,
            self.test_corrupted_files,
            self.test_query_edge_cases,
            self.test_concurrent_operations,
            self.test_memory_persistence,
            self.test_large_query_results,
        ]
        
        # Run all tests
        results = []
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
                
                # Print result
                status = result["status"]
                test_name = result["test"]
                message = result["message"]
                
                status_symbol = {
                    "PASS": "✓",
                    "FAIL": "✗",
                    "ERROR": "💥",
                    "PARTIAL": "⚠"
                }.get(status, "?")
                
                print(f"{status_symbol} {test_name:20} | {message}")
                
            except Exception as e:
                error_result = {"test": test_method.__name__, "status": "ERROR", "message": str(e)}
                results.append(error_result)
                print(f"💥 {test_method.__name__:20} | Exception: {e}")
        
        # Generate summary
        status_counts = {}
        for result in results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_tests = len(results)
        passed_tests = status_counts.get("PASS", 0)
        
        print(f"\n{'='*40}")
        print("EDGE CASE TEST SUMMARY:")
        print(f"Total Tests: {total_tests}")
        for status, count in status_counts.items():
            print(f"{status}: {count}")
        
        return {
            "total_tests": total_tests,
            "results": results,
            "status_counts": status_counts,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0
        }


async def test_real_world_scenario() -> Dict[str, Any]:
    """Test a real-world scenario using the actual project codebase."""
    print("🌍 Testing Real-World Scenario: Self-Indexing")
    print("=" * 50)
    
    client = MockMCPClient()
    
    # Use the actual project directory
    project_path = "/Users/rickbarraza/dev/mcp/mcp-server-repo-memory"
    project_name = "mcp-server-repo-memory-self-test"
    
    try:
        # Index the actual project
        print("Indexing the actual project codebase...")
        start_time = time.time()
        
        result = await client.call_tool('index_project_memory', {
            'project_path': project_path,
            'project_name': project_name,
            'force_reindex': True
        })
        
        index_time = time.time() - start_time
        result_data = json.loads(result)
        
        if not result_data.get("success"):
            return {"status": "FAIL", "message": f"Self-indexing failed: {result_data.get('error')}"}
        
        chunks_indexed = result_data.get("chunks_indexed", 0)
        print(f"✓ Indexed {chunks_indexed} chunks in {index_time:.2f}s")
        
        # Test realistic queries about the codebase
        realistic_queries = [
            "project memory service implementation",
            "embedding service with ChromaDB",
            "MCP tool registration for memory",
            "code chunk extraction from files",
            "semantic search functionality",
            "error handling in memory operations",
            "async method for indexing projects",
            "configuration management for services",
            "logging setup and initialization",
            "test fixtures and mocking utilities"
        ]
        
        query_results = []
        print("\nTesting realistic queries...")
        
        for query in realistic_queries:
            try:
                start = time.time()
                result = await client.call_tool('query_project_memory', {
                    'project_name': project_name,
                    'query': query,
                    'context_limit': 3,
                    'include_metadata': True
                })
                elapsed = (time.time() - start) * 1000
                
                # Check if we got reasonable results
                has_results = "Found" in result and "code chunks" in result
                result_length = len(result)
                
                query_results.append({
                    "query": query,
                    "has_results": has_results,
                    "response_time_ms": elapsed,
                    "response_length": result_length
                })
                
                status = "✓" if has_results else "✗"
                print(f"  {status} {query[:30]:30} | {elapsed:.1f}ms | {result_length} chars")
                
            except Exception as e:
                query_results.append({
                    "query": query,
                    "has_results": False,
                    "error": str(e),
                    "response_time_ms": 0,
                    "response_length": 0
                })
                print(f"  ✗ {query[:30]:30} | ERROR: {e}")
        
        # Analyze results
        successful_queries = [r for r in query_results if r.get("has_results", False)]
        success_rate = len(successful_queries) / len(query_results)
        avg_response_time = sum(r.get("response_time_ms", 0) for r in successful_queries) / len(successful_queries) if successful_queries else 0
        
        print(f"\n📊 Real-World Test Results:")
        print(f"Chunks Indexed: {chunks_indexed}")
        print(f"Index Time: {index_time:.2f}s")
        print(f"Query Success Rate: {success_rate:.1%} ({len(successful_queries)}/{len(query_results)})")
        print(f"Average Query Time: {avg_response_time:.1f}ms")
        
        if success_rate >= 0.8 and avg_response_time < 50:
            return {"status": "PASS", "message": f"Real-world test successful: {success_rate:.1%} success rate, {avg_response_time:.1f}ms avg"}
        elif success_rate >= 0.6:
            return {"status": "PARTIAL", "message": f"Partial success: {success_rate:.1%} success rate"}
        else:
            return {"status": "FAIL", "message": f"Poor performance: {success_rate:.1%} success rate"}
            
    except Exception as e:
        return {"status": "ERROR", "message": f"Real-world test failed: {e}"}


async def main():
    """Main entry point for comprehensive testing."""
    print("🚀 Starting Comprehensive Test Suite - Milestone 4")
    print("=" * 60)
    
    all_results = []
    
    try:
        # Run edge case tests
        edge_tests = EdgeCaseTests()
        edge_results = await edge_tests.run_all_edge_case_tests()
        all_results.append(("Edge Case Tests", edge_results))
        
        print("\n" + "="*60)
        
        # Run real-world scenario test
        real_world_result = await test_real_world_scenario()
        all_results.append(("Real-World Scenario", real_world_result))
        
        # Final summary
        print(f"\n{'='*60}")
        print("MILESTONE 4 - COMPREHENSIVE TEST SUMMARY")
        print("-" * 60)
        
        for test_name, result in all_results:
            if isinstance(result, dict) and "success_rate" in result:
                success_rate = result["success_rate"]
                total_tests = result["total_tests"]
                print(f"{test_name:25} | {success_rate:.1%} success ({total_tests} tests)")
            elif isinstance(result, dict) and "status" in result:
                status = result["status"]
                message = result.get("message", "")
                print(f"{test_name:25} | {status} - {message}")
        
        print(f"\n🎉 Milestone 4 Integration & Testing COMPLETE!")
        return True
        
    except Exception as e:
        print(f"\n❌ Comprehensive testing failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
