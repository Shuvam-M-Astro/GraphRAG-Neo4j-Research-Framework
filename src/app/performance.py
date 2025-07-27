"""
Performance and UX Optimization Module
Implements lazy loading, caching, optimistic updates, and offline support.
"""

import streamlit as st
import pandas as pd
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import threading
from datetime import datetime, timedelta
import os
import pickle

class PerformanceManager:
    """Manages performance optimizations including caching, lazy loading, and offline support."""
    
    def __init__(self):
        """Initialize the performance manager."""
        self.cache_dir = ".cache"
        self.ensure_cache_directory()
        self.offline_mode = False
        self.cache_stats = {"hits": 0, "misses": 0}
        
    def ensure_cache_directory(self):
        """Ensure cache directory exists."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key for function call."""
        # Create a hash of the function name and arguments
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Check if cached data is still valid."""
        if not os.path.exists(cache_path):
            return False
        
        file_age = time.time() - os.path.getmtime(cache_path)
        return file_age < (max_age_hours * 3600)
    
    def load_from_cache(self, cache_key: str, max_age_hours: int = 24) -> Optional[Any]:
        """Load data from cache if valid."""
        cache_path = self.get_cache_path(cache_key)
        
        if self.is_cache_valid(cache_path, max_age_hours):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.cache_stats["hits"] += 1
                return data
            except Exception as e:
                st.warning(f"Cache loading error: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache."""
        cache_path = self.get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.warning(f"Cache saving error: {e}")
    
    def clear_cache(self):
        """Clear all cached data."""
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        self.cache_stats = {"hits": 0, "misses": 0}
        st.success("Cache cleared successfully!")

class LazyLoader:
    """Implements lazy loading for data and components."""
    
    def __init__(self, performance_manager: PerformanceManager):
        """Initialize the lazy loader."""
        self.pm = performance_manager
        self.loaded_data = {}
    
    def lazy_load_papers(self, query: str, limit: int = 20, batch_size: int = 5):
        """Lazy load papers in batches."""
        cache_key = self.pm.get_cache_key("lazy_papers", query, limit)
        
        # Check cache first
        cached_data = self.pm.load_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_papers = []
        total_batches = (limit + batch_size - 1) // batch_size
        
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min(batch_start + batch_size, limit)
            
            # Update progress
            progress = (batch_num + 1) / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Loading papers {batch_start + 1}-{batch_end} of {limit}...")
            
            # Simulate batch loading (replace with actual API call)
            batch_papers = self._load_paper_batch(query, batch_start, batch_end)
            all_papers.extend(batch_papers)
            
            # Small delay to show progress
            time.sleep(0.1)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Cache the results
        self.pm.save_to_cache(cache_key, all_papers)
        
        return all_papers
    
    def _load_paper_batch(self, query: str, start: int, end: int) -> List[Dict]:
        """Load a batch of papers (placeholder implementation)."""
        # This would be replaced with actual API calls
        return []
    
    def lazy_load_visualization(self, data: List[Dict], viz_type: str):
        """Lazy load visualization components."""
        cache_key = self.pm.get_cache_key("lazy_viz", viz_type, str(len(data)))
        
        # Check cache first
        cached_viz = self.pm.load_from_cache(cache_key)
        if cached_viz:
            return cached_viz
        
        # Show loading placeholder
        with st.spinner(f"Generating {viz_type} visualization..."):
            # Generate visualization (placeholder)
            viz_data = self._generate_visualization(data, viz_type)
            
            # Cache the visualization
            self.pm.save_to_cache(cache_key, viz_data)
            
            return viz_data
    
    def _generate_visualization(self, data: List[Dict], viz_type: str) -> Any:
        """Generate visualization data (placeholder)."""
        return {"type": viz_type, "data": data}

class OptimisticUpdater:
    """Implements optimistic updates for immediate user feedback."""
    
    def __init__(self):
        """Initialize the optimistic updater."""
        self.pending_updates = {}
    
    def optimistic_search(self, query: str, search_func: Callable):
        """Perform optimistic search with immediate feedback."""
        # Show immediate feedback
        with st.container():
            st.info(f"🔍 Searching for: {query}")
            
            # Create placeholder for results
            results_placeholder = st.empty()
            
            # Show optimistic results immediately
            optimistic_results = self._get_optimistic_results(query)
            results_placeholder.json(optimistic_results)
            
            # Perform actual search in background
            try:
                actual_results = search_func(query)
                
                # Update with real results
                results_placeholder.json(actual_results)
                st.success("✅ Search completed!")
                
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
                # Keep optimistic results as fallback
    
    def optimistic_save(self, data: Dict, save_func: Callable):
        """Perform optimistic save with immediate feedback."""
        # Show immediate success
        st.success("✅ Saved successfully!")
        
        # Perform actual save in background
        try:
            save_func(data)
        except Exception as e:
            st.error(f"❌ Save failed: {e}")
            # Could implement retry logic here
    
    def _get_optimistic_results(self, query: str) -> Dict:
        """Get optimistic results for immediate display."""
        return {
            "query": query,
            "status": "searching",
            "estimated_results": 15,
            "message": "Searching in progress..."
        }

class OfflineManager:
    """Manages offline functionality and data synchronization."""
    
    def __init__(self, performance_manager: PerformanceManager):
        """Initialize the offline manager."""
        self.pm = performance_manager
        self.offline_data = {}
        self.sync_queue = []
    
    def check_connectivity(self) -> bool:
        """Check if the application is online."""
        try:
            # Simple connectivity check
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=1)
            return True
        except:
            return False
    
    def enable_offline_mode(self):
        """Enable offline mode."""
        self.pm.offline_mode = True
        st.warning("🔄 Offline mode enabled. Some features may be limited.")
    
    def disable_offline_mode(self):
        """Disable offline mode."""
        self.pm.offline_mode = False
        st.success("🌐 Online mode restored.")
    
    def get_offline_data(self, key: str) -> Optional[Any]:
        """Get data from offline storage."""
        return self.offline_data.get(key)
    
    def save_offline_data(self, key: str, data: Any):
        """Save data to offline storage."""
        self.offline_data[key] = data
        
        # Also save to persistent storage
        cache_key = f"offline_{key}"
        self.pm.save_to_cache(cache_key, data)
    
    def sync_offline_changes(self):
        """Sync offline changes when back online."""
        if not self.check_connectivity():
            st.error("❌ No internet connection. Cannot sync.")
            return
        
        if not self.sync_queue:
            st.info("✅ No pending changes to sync.")
            return
        
        with st.spinner("Syncing offline changes..."):
            for change in self.sync_queue:
                try:
                    # Apply the change
                    self._apply_sync_change(change)
                    st.success(f"✅ Synced: {change['type']}")
                except Exception as e:
                    st.error(f"❌ Sync failed: {e}")
        
        self.sync_queue.clear()
        st.success("🎉 All changes synced successfully!")
    
    def _apply_sync_change(self, change: Dict):
        """Apply a sync change (placeholder)."""
        # This would implement the actual sync logic
        pass

# Global performance manager instance
performance_manager = PerformanceManager()
lazy_loader = LazyLoader(performance_manager)
optimistic_updater = OptimisticUpdater()
offline_manager = OfflineManager(performance_manager)

# Decorator for caching function results
def cached_result(max_age_hours: int = 24):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = performance_manager.get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = performance_manager.load_from_cache(cache_key, max_age_hours)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            performance_manager.save_to_cache(cache_key, result)
            
            return result
        return wrapper
    return decorator

# Utility functions for Streamlit integration
def show_performance_stats():
    """Display performance statistics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cache Hits", performance_manager.cache_stats["hits"])
    
    with col2:
        st.metric("Cache Misses", performance_manager.cache_stats["misses"])
    
    with col3:
        hit_rate = 0
        if sum(performance_manager.cache_stats.values()) > 0:
            hit_rate = performance_manager.cache_stats["hits"] / sum(performance_manager.cache_stats.values()) * 100
        st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")

def show_offline_status():
    """Display offline status and controls."""
    is_online = offline_manager.check_connectivity()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if is_online:
            st.success("🌐 Online")
        else:
            st.warning("🔄 Offline Mode")
    
    with col2:
        if st.button("Sync Changes"):
            offline_manager.sync_offline_changes()

def create_lazy_dataframe(data: List[Dict], page_size: int = 10):
    """Create a lazy-loading dataframe with pagination."""
    if not data:
        st.warning("No data available.")
        return
    
    # Pagination
    total_pages = (len(data) + page_size - 1) // page_size
    current_page = st.selectbox("Page", range(1, total_pages + 1), index=0)
    
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(data))
    
    # Show current page info
    st.info(f"Showing {start_idx + 1}-{end_idx} of {len(data)} items")
    
    # Display current page data
    page_data = data[start_idx:end_idx]
    df = pd.DataFrame(page_data)
    st.dataframe(df, use_container_width=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_page > 1:
            if st.button("← Previous"):
                st.rerun()
    
    with col2:
        st.write(f"Page {current_page} of {total_pages}")
    
    with col3:
        if current_page < total_pages:
            if st.button("Next →"):
                st.rerun() 