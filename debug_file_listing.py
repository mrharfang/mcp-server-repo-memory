#!/usr/bin/env python3
"""Debug script to check what files are being found."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_server_tree_sitter.tools.file_operations import list_project_files
from mcp_server_tree_sitter.api import register_project, get_project_registry

def debug_file_listing():
    """Debug what files are being found."""
    print("🔍 Debugging file listing...")
    
    project_path = Path(__file__).parent.absolute()
    project_name = "debug-project"
    
    # Register project
    register_result = register_project(str(project_path), project_name)
    print(f"📁 Registered project: {register_result}")
    
    # Get project from registry
    registry = get_project_registry()
    project_obj = registry.get_project(project_name)
    
    # Test with Python extensions
    code_extensions_list = ['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'go', 'rs', 'swift']
    python_files = list_project_files(project_obj, filter_extensions=['py'])
    all_code_files = list_project_files(project_obj, filter_extensions=code_extensions_list)
    all_files = list_project_files(project_obj)
    
    print(f"📊 Results:")
    print(f"   • Python files (.py): {len(python_files)}")
    print(f"   • All code files: {len(all_code_files)}")
    print(f"   • All files: {len(all_files)}")
    
    print(f"\n📝 First 10 Python files:")
    for i, file_path in enumerate(python_files[:10]):
        print(f"   {i+1}. {file_path}")
    
    print(f"\n📝 Sample of all code files (first 10):")
    for i, file_path in enumerate(all_code_files[:10]):
        print(f"   {i+1}. {file_path}")

if __name__ == "__main__":
    debug_file_listing()
