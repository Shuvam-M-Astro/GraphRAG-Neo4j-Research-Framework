#!/usr/bin/env python3
"""
Test script to debug import issues
"""

import sys
import os

# Add src to path
sys.path.append('src')

print("Testing imports...")

try:
    print("1. Testing basic module import...")
    import graph_rag
    print("   ✓ graph_rag module imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import graph_rag: {e}")

try:
    print("2. Testing orchestrator import...")
    from graph_rag.orchestrator import GraphRAGOrchestrator
    print("   ✓ GraphRAGOrchestrator imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import GraphRAGOrchestrator: {e}")

try:
    print("3. Testing direct file import...")
    import graph_rag.orchestrator as orchestrator_module
    print("   ✓ orchestrator module imported successfully")
    print(f"   Available names: {dir(orchestrator_module)}")
except Exception as e:
    print(f"   ✗ Failed to import orchestrator module: {e}")

try:
    print("4. Testing class instantiation...")
    orchestrator = GraphRAGOrchestrator()
    print("   ✓ GraphRAGOrchestrator instantiated successfully")
except Exception as e:
    print(f"   ✗ Failed to instantiate GraphRAGOrchestrator: {e}")

print("Import test completed.") 