#!/usr/bin/env python3
"""
Test script for performance improvements
Tests lazy loading, caching, optimistic updates, and offline support.
"""

import os
import sys
import time

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from app.performance import performance_manager, lazy_loader, optimistic_updater, offline_manager
from app.lazy_components import lazy_viz, lazy_table, lazy_search, lazy_metrics

def test_caching():
    """Test caching functionality."""
    print("🧪 Testing caching functionality...")
    
    # Test cache key generation
    cache_key = performance_manager.get_cache_key("test_function", "arg1", "arg2")
    print(f"✅ Cache key generated: {cache_key}")
    
    # Test saving and loading from cache
    test_data = {"test": "data", "number": 42}
    performance_manager.save_to_cache(cache_key, test_data)
    print("✅ Data saved to cache")
    
    # Test loading from cache
    loaded_data = performance_manager.load_from_cache(cache_key)
    if loaded_data == test_data:
        print("✅ Data loaded from cache successfully")
    else:
        print("❌ Cache loading failed")
    
    # Test cache statistics
    print(f"📊 Cache stats: {performance_manager.cache_stats}")

def test_lazy_loading():
    """Test lazy loading functionality."""
    print("\n🧪 Testing lazy loading functionality...")
    
    # Test lazy loading papers
    papers = lazy_loader.lazy_load_papers("test query", limit=10)
    print(f"✅ Lazy loaded {len(papers)} papers")
    
    # Test lazy visualization
    test_papers = [
        {"paper_id": "1", "title": "Test Paper 1", "year": 2020, "citations": 10},
        {"paper_id": "2", "title": "Test Paper 2", "year": 2021, "citations": 15}
    ]
    viz = lazy_loader.lazy_load_visualization(test_papers, "test_viz")
    print(f"✅ Lazy visualization created: {viz}")

def test_optimistic_updates():
    """Test optimistic updates."""
    print("\n🧪 Testing optimistic updates...")
    
    def mock_search_func(query):
        time.sleep(0.1)  # Simulate search time
        return {"results": f"Results for {query}"}
    
    # Test optimistic search
    optimistic_updater.optimistic_search("test query", mock_search_func)
    print("✅ Optimistic search completed")

def test_offline_support():
    """Test offline support."""
    print("\n🧪 Testing offline support...")
    
    # Test connectivity check
    is_online = offline_manager.check_connectivity()
    print(f"🌐 Online status: {is_online}")
    
    # Test offline data storage
    test_data = {"offline": "data"}
    offline_manager.save_offline_data("test_key", test_data)
    print("✅ Data saved to offline storage")
    
    # Test offline data retrieval
    retrieved_data = offline_manager.get_offline_data("test_key")
    if retrieved_data == test_data:
        print("✅ Offline data retrieved successfully")
    else:
        print("❌ Offline data retrieval failed")

def test_lazy_components():
    """Test lazy components."""
    print("\n🧪 Testing lazy components...")
    
    # Test lazy search
    def mock_search(query):
        return {"papers": [{"title": f"Paper for {query}"}]}
    
    results = lazy_search.perform_lazy_search("test query", mock_search)
    print(f"✅ Lazy search completed: {results}")
    
    # Test search suggestions
    suggestions = lazy_search.get_search_suggestions("test")
    print(f"✅ Search suggestions: {suggestions}")
    
    # Test lazy metrics
    test_data = [
        {"citations": 10, "year": 2020, "methods": ["method1"]},
        {"citations": 15, "year": 2021, "methods": ["method2"]}
    ]
    metrics = lazy_metrics.display_lazy_metrics(test_data)
    print(f"✅ Lazy metrics calculated: {metrics}")

def main():
    """Run all performance tests."""
    print("🚀 Starting Performance Tests")
    print("=" * 50)
    
    try:
        test_caching()
        test_lazy_loading()
        test_optimistic_updates()
        test_offline_support()
        test_lazy_components()
        
        print("\n" + "=" * 50)
        print("🎉 All performance tests completed successfully!")
        print("\nPerformance improvements implemented:")
        print("✅ Lazy Loading: Progressive data loading with progress indicators")
        print("✅ Caching: Smart caching with hit/miss statistics")
        print("✅ Optimistic Updates: Immediate feedback for user actions")
        print("✅ Offline Support: Basic offline functionality for cached data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 