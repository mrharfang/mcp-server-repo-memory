"""
Performance testing module for Project Memory MCP
Tests memory system performance with larger codebases and stress conditions.
"""

import asyncio
import tempfile
import shutil
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from integration_test import MockMCPClient, IntegrationTestError


class PerformanceTests:
    """Performance test suite for Project Memory MCP."""
    
    def __init__(self):
        """Initialize performance test suite."""
        self.client = None
        self.results = []
    
    async def setup(self):
        """Set up the test environment."""
        self.client = MockMCPClient()
        print("✓ Performance test setup complete")
    
    async def create_large_codebase(self, num_files: int = 50, lines_per_file: int = 100) -> str:
        """Create a larger synthetic codebase for performance testing."""
        temp_dir = tempfile.mkdtemp(prefix="perf_test_codebase_")
        print(f"Creating large test codebase: {num_files} files, ~{lines_per_file} lines each")
        
        for i in range(num_files):
            file_content = f'''"""
Module {i}: Synthetic code for performance testing.
Generated file with realistic Python patterns.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class DataModel{i}:
    """Data model for module {i}."""
    
    id: int
    name: str
    value: float
    metadata: Dict[str, Any]
    active: bool = True
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.id < 0:
            raise ValueError("ID must be non-negative")
        if not self.name:
            raise ValueError("Name cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {{
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata,
            "active": self.active
        }}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataModel{i}":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            value=data["value"],
            metadata=data["metadata"],
            active=data.get("active", True)
        )


class Service{i}:
    """Service class for module {i} operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize service with configuration."""
        self.config = config or {{}}
        self.data_store = []
        self.cache = {{}}
        self.initialized = False
        
        logger.info(f"Initialized Service{i} with config: {{self.config}}")
    
    async def initialize(self) -> None:
        """Initialize the service asynchronously."""
        try:
            # Simulate async initialization
            await asyncio.sleep(0.001)
            
            # Load default data
            await self.load_default_data()
            
            self.initialized = True
            logger.info(f"Service{i} initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize Service{i}: {{e}}")
            raise
    
    async def load_default_data(self) -> None:
        """Load default data into the service."""
        default_items = [
            DataModel{i}(
                id=j,
                name=f"item_{{j}}",
                value=float(j * 1.5),
                metadata={{"type": "default", "module": {i}}}
            )
            for j in range(10)
        ]
        
        self.data_store.extend(default_items)
        logger.debug(f"Loaded {{len(default_items)}} default items")
    
    def create_item(self, name: str, value: float, 
                   metadata: Optional[Dict[str, Any]] = None) -> DataModel{i}:
        """Create a new data item."""
        if not self.initialized:
            raise RuntimeError("Service not initialized")
        
        item_id = len(self.data_store)
        item = DataModel{i}(
            id=item_id,
            name=name,
            value=value,
            metadata=metadata or {{}}
        )
        
        self.data_store.append(item)
        
        # Update cache
        self.cache[item_id] = item
        
        logger.debug(f"Created item {{item_id}}: {{name}}")
        return item
    
    def get_item(self, item_id: int) -> Optional[DataModel{i}]:
        """Get item by ID with caching."""
        if item_id in self.cache:
            return self.cache[item_id]
        
        for item in self.data_store:
            if item.id == item_id:
                self.cache[item_id] = item
                return item
        
        return None
    
    def list_items(self, active_only: bool = True) -> List[DataModel{i}]:
        """List all items, optionally filtering by active status."""
        if active_only:
            return [item for item in self.data_store if item.active]
        return self.data_store.copy()
    
    def update_item(self, item_id: int, **kwargs) -> bool:
        """Update an existing item."""
        item = self.get_item(item_id)
        if not item:
            return False
        
        for key, value in kwargs.items():
            if hasattr(item, key):
                setattr(item, key, value)
        
        # Invalidate cache
        if item_id in self.cache:
            del self.cache[item_id]
        
        logger.debug(f"Updated item {{item_id}}")
        return True
    
    def delete_item(self, item_id: int) -> bool:
        """Soft delete an item by marking it as inactive."""
        return self.update_item(item_id, active=False)
    
    def search_items(self, query: str) -> List[DataModel{i}]:
        """Search items by name containing query."""
        results = []
        query_lower = query.lower()
        
        for item in self.data_store:
            if query_lower in item.name.lower() and item.active:
                results.append(item)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        active_items = [item for item in self.data_store if item.active]
        
        values = [item.value for item in active_items]
        
        return {{
            "total_items": len(self.data_store),
            "active_items": len(active_items),
            "average_value": sum(values) / len(values) if values else 0,
            "min_value": min(values) if values else 0,
            "max_value": max(values) if values else 0,
            "cache_size": len(self.cache)
        }}
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self.cache.clear()
        logger.debug("Cache cleared")
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        try:
            self.clear_cache()
            self.data_store.clear()
            self.initialized = False
            
            logger.info(f"Service{i} cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Service{i} cleanup: {{e}}")


def process_batch(items: List[DataModel{i}], 
                 operation: str = "validate") -> List[Dict[str, Any]]:
    """Process a batch of items with the specified operation."""
    results = []
    
    for item in items:
        try:
            if operation == "validate":
                # Validation logic
                is_valid = (
                    item.id >= 0 and
                    len(item.name) > 0 and
                    item.value is not None
                )
                results.append({{"id": item.id, "valid": is_valid}})
                
            elif operation == "transform":
                # Transform item data
                transformed = {{
                    "id": item.id,
                    "display_name": item.name.upper(),
                    "formatted_value": f"${{item.value:.2f}}",
                    "status": "active" if item.active else "inactive"
                }}
                results.append(transformed)
                
            elif operation == "analyze":
                # Analyze item properties
                analysis = {{
                    "id": item.id,
                    "name_length": len(item.name),
                    "value_category": "high" if item.value > 50 else "low",
                    "metadata_count": len(item.metadata)
                }}
                results.append(analysis)
                
        except Exception as e:
            results.append({{"id": getattr(item, "id", -1), "error": str(e)}})
    
    return results


# Module-level constants and utilities
MODULE_VERSION = "1.{i}.0"
DEFAULT_CONFIG = {{
    "batch_size": 100,
    "cache_enabled": True,
    "log_level": "INFO",
    "timeout": 30
}}

def get_module_info() -> Dict[str, Any]:
    """Get information about this module."""
    return {{
        "module_id": {i},
        "version": MODULE_VERSION,
        "classes": ["DataModel{i}", "Service{i}"],
        "functions": ["process_batch", "get_module_info"],
        "config": DEFAULT_CONFIG
    }}
'''
            
            file_path = Path(temp_dir) / f"module_{i:03d}.py"
            file_path.write_text(file_content, encoding='utf-8')
        
        # Create a main module that imports everything
        main_content = f'''"""
Main module for large codebase performance test.
Imports and uses all {num_files} generated modules.
"""

import asyncio
import logging
from typing import List, Dict, Any

# Import all generated modules
{chr(10).join([f"from module_{i:03d} import Service{i}, DataModel{i}" for i in range(num_files)])}

logger = logging.getLogger(__name__)


async def initialize_all_services() -> List[Any]:
    """Initialize all services concurrently."""
    services = []
    
    for i in range({num_files}):
        service_class = globals()[f"Service{{i}}"]
        service = service_class({{"module_id": i}})
        await service.initialize()
        services.append(service)
    
    logger.info(f"Initialized {{len(services)}} services")
    return services


async def process_all_data(services: List[Any]) -> Dict[str, Any]:
    """Process data across all services."""
    total_items = 0
    total_value = 0.0
    
    for i, service in enumerate(services):
        items = service.list_items()
        total_items += len(items)
        total_value += sum(item.value for item in items)
        
        # Create some additional items
        for j in range(5):
            service.create_item(
                name=f"generated_{{i}}_{{j}}",
                value=float(i * 10 + j),
                metadata={{"generated": True, "service": i}}
            )
    
    return {{
        "total_services": len(services),
        "total_items": total_items,
        "total_value": total_value,
        "average_value_per_service": total_value / len(services) if services else 0
    }}


if __name__ == "__main__":
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        services = await initialize_all_services()
        results = await process_all_data(services)
        
        print(f"Processing complete: {{results}}")
        
        # Cleanup
        for service in services:
            await service.cleanup()
    
    asyncio.run(main())
'''
        
        main_path = Path(temp_dir) / "main.py"
        main_path.write_text(main_content, encoding='utf-8')
        
        print(f"✓ Created large codebase with {num_files + 1} files")
        return temp_dir
    
    async def test_indexing_performance(self, codebase_path: str, project_name: str) -> Dict[str, Any]:
        """Test indexing performance with timing metrics."""
        print(f"Testing indexing performance for {project_name}...")
        
        start_time = time.time()
        
        try:
            result = await self.client.call_tool('index_project_memory', {
                'project_path': codebase_path,
                'project_name': project_name,
                'force_reindex': True
            })
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Parse result
            import json
            result_data = json.loads(result)
            
            if result_data.get("success"):
                chunks_indexed = result_data.get("chunks_indexed", 0)
                throughput = chunks_indexed / elapsed if elapsed > 0 else 0
                
                metrics = {
                    "success": True,
                    "elapsed_time": elapsed,
                    "chunks_indexed": chunks_indexed,
                    "throughput_chunks_per_sec": throughput,
                    "project_name": project_name
                }
                
                print(f"  ✓ Indexed {chunks_indexed} chunks in {elapsed:.2f}s ({throughput:.1f} chunks/sec)")
                return metrics
            else:
                error = result_data.get("error", "Unknown error")
                print(f"  ✗ Indexing failed: {error}")
                return {"success": False, "error": error}
                
        except Exception as e:
            print(f"  ✗ Performance test failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def test_query_performance(self, project_name: str, 
                                   queries: List[str], iterations: int = 5) -> Dict[str, Any]:
        """Test query performance with multiple iterations."""
        print(f"Testing query performance for {project_name} with {len(queries)} queries...")
        
        all_times = []
        query_results = {}
        
        for query in queries:
            query_times = []
            
            for iteration in range(iterations):
                start_time = time.time()
                
                try:
                    result = await self.client.call_tool('query_project_memory', {
                        'project_name': project_name,
                        'query': query,
                        'context_limit': 5,
                        'include_metadata': True
                    })
                    
                    end_time = time.time()
                    elapsed = (end_time - start_time) * 1000  # Convert to ms
                    query_times.append(elapsed)
                    all_times.append(elapsed)
                    
                except Exception as e:
                    print(f"    ✗ Query failed: {e}")
                    query_times.append(float('inf'))
            
            # Calculate statistics for this query
            valid_times = [t for t in query_times if t != float('inf')]
            if valid_times:
                query_results[query] = {
                    "avg_time_ms": statistics.mean(valid_times),
                    "min_time_ms": min(valid_times),
                    "max_time_ms": max(valid_times),
                    "median_time_ms": statistics.median(valid_times),
                    "success_rate": len(valid_times) / iterations
                }
                
                avg_time = statistics.mean(valid_times)
                print(f"  Query: '{query[:30]}...' avg: {avg_time:.1f}ms")
        
        # Overall statistics
        valid_all_times = [t for t in all_times if t != float('inf')]
        
        return {
            "total_queries": len(queries),
            "iterations_per_query": iterations,
            "query_details": query_results,
            "overall_avg_time_ms": statistics.mean(valid_all_times) if valid_all_times else 0,
            "overall_median_time_ms": statistics.median(valid_all_times) if valid_all_times else 0,
            "total_success_rate": len(valid_all_times) / len(all_times) if all_times else 0
        }
    
    async def test_memory_scalability(self) -> List[Dict[str, Any]]:
        """Test how memory system scales with different codebase sizes."""
        print("Testing memory system scalability...")
        
        test_sizes = [
            (10, "small"),
            (25, "medium"), 
            (50, "large")
        ]
        
        scalability_results = []
        
        for num_files, size_label in test_sizes:
            print(f"\n  Testing {size_label} codebase ({num_files} files)...")
            
            # Create codebase
            codebase_path = await self.create_large_codebase(num_files=num_files)
            project_name = f"scalability-test-{size_label}"
            
            try:
                # Test indexing performance
                index_metrics = await self.test_indexing_performance(codebase_path, project_name)
                
                if index_metrics.get("success"):
                    # Test query performance
                    test_queries = [
                        "service initialization",
                        "data model validation", 
                        "batch processing",
                        "error handling",
                        "async operations"
                    ]
                    
                    query_metrics = await self.test_query_performance(project_name, test_queries, iterations=3)
                    
                    scalability_results.append({
                        "size_label": size_label,
                        "num_files": num_files,
                        "indexing": index_metrics,
                        "querying": query_metrics
                    })
                    
                    print(f"    ✓ {size_label} test completed")
                else:
                    print(f"    ✗ {size_label} test failed during indexing")
                
            finally:
                # Cleanup
                shutil.rmtree(codebase_path, ignore_errors=True)
        
        return scalability_results
    
    async def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run the complete performance test suite."""
        print("🚀 Starting Performance Test Suite")
        print("=" * 50)
        
        await self.setup()
        
        # Test 1: Scalability testing
        scalability_results = await self.test_memory_scalability()
        
        # Generate performance report
        report = {
            "timestamp": time.time(),
            "scalability_tests": scalability_results,
            "summary": self._generate_performance_summary(scalability_results)
        }
        
        return report
    
    def _generate_performance_summary(self, scalability_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of performance test results."""
        if not scalability_results:
            return {"status": "no_results"}
        
        # Extract key metrics
        indexing_times = []
        query_times = []
        throughputs = []
        
        for result in scalability_results:
            if result.get("indexing", {}).get("success"):
                indexing_times.append(result["indexing"]["elapsed_time"])
                throughputs.append(result["indexing"]["throughput_chunks_per_sec"])
            
            if result.get("querying"):
                query_times.append(result["querying"]["overall_avg_time_ms"])
        
        # Calculate summary statistics
        summary = {
            "tests_completed": len(scalability_results),
            "indexing_performance": {
                "avg_time_seconds": statistics.mean(indexing_times) if indexing_times else 0,
                "avg_throughput_chunks_per_sec": statistics.mean(throughputs) if throughputs else 0
            },
            "query_performance": {
                "avg_time_ms": statistics.mean(query_times) if query_times else 0
            },
            "recommendations": self._generate_recommendations(scalability_results)
        }
        
        return summary
    
    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Check indexing performance
        indexing_times = [r["indexing"]["elapsed_time"] for r in results 
                         if r.get("indexing", {}).get("success")]
        
        if indexing_times:
            max_time = max(indexing_times)
            if max_time > 10:  # More than 10 seconds for indexing
                recommendations.append("Consider implementing parallel indexing for large codebases")
            
            # Check if performance degrades with size
            if len(indexing_times) > 1:
                if indexing_times[-1] / indexing_times[0] > 5:  # 5x slower for largest
                    recommendations.append("Indexing performance degrades significantly with size")
        
        # Check query performance
        query_times = [r["querying"]["overall_avg_time_ms"] for r in results 
                      if r.get("querying")]
        
        if query_times:
            max_query_time = max(query_times)
            if max_query_time > 1000:  # More than 1 second
                recommendations.append("Query performance may be too slow for interactive use")
        
        if not recommendations:
            recommendations.append("Performance looks good across all test sizes")
        
        return recommendations


async def main():
    """Main entry point for performance testing."""
    perf_tests = PerformanceTests()
    
    try:
        results = await perf_tests.run_all_performance_tests()
        
        # Print results
        print(f"\n{'='*50}")
        print("PERFORMANCE TEST RESULTS:")
        print("-" * 30)
        
        summary = results["summary"]
        print(f"Tests Completed: {summary['tests_completed']}")
        print(f"Avg Indexing Time: {summary['indexing_performance']['avg_time_seconds']:.2f}s")
        print(f"Avg Throughput: {summary['indexing_performance']['avg_throughput_chunks_per_sec']:.1f} chunks/sec")
        print(f"Avg Query Time: {summary['query_performance']['avg_time_ms']:.1f}ms")
        
        print("\nRecommendations:")
        for rec in summary["recommendations"]:
            print(f"  • {rec}")
        
        print(f"\n🎯 Performance testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
