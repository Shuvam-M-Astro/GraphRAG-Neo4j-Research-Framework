#!/usr/bin/env python3
"""
Complete System Setup for GraphRAG Neo4j Research Framework
Initializes database, creates sample data, and builds vector index.
"""

import os
import sys
import logging
import subprocess
from dotenv import load_dotenv

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from database.init_database import Neo4jDatabase
from graph_rag.graph_retriever import GraphRetriever

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_neo4j_connection():
    """Check if Neo4j is accessible."""
    try:
        db = Neo4jDatabase()
        db.connect()
        db.close()
        logger.info("✅ Neo4j connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        logger.error("Please ensure Neo4j is running on localhost:7687")
        return False

def initialize_database():
    """Initialize the Neo4j database with schema and sample data."""
    try:
        logger.info("Initializing Neo4j database...")
        db = Neo4jDatabase()
        db.connect()
        db.create_constraints_and_indexes()
        db.create_sample_data()
        db.close()
        logger.info("✅ Database initialization completed")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def build_vector_index():
    """Build the vector index for hybrid search."""
    try:
        logger.info("Building vector index...")
        retriever = GraphRetriever()
        
        if retriever.documents and retriever.document_metadata:
            logger.info(f"✅ Vector index built successfully!")
            logger.info(f"   - Documents indexed: {len(retriever.documents)}")
            logger.info(f"   - Index build time: {retriever.index_build_time:.2f}s")
            
            # Test the index
            logger.info("Testing vector index...")
            results = retriever.hybrid_search("deep learning", limit=3)
            logger.info(f"✅ Test search successful! Found {len(results)} results")
            
            retriever.close()
            return True
        else:
            logger.error("❌ Vector index build failed - no documents found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector index build failed: {e}")
        return False

def test_system():
    """Run comprehensive system tests."""
    try:
        logger.info("Running system tests...")
        
        # Test GraphRetriever
        retriever = GraphRetriever()
        
        # Test different search types
        test_queries = [
            ("deep learning", "hybrid"),
            ("graph neural networks", "graph"),
            ("transformer", "vector")
        ]
        
        for query, search_type in test_queries:
            try:
                if search_type == "hybrid":
                    results = retriever.hybrid_search(query, limit=3)
                elif search_type == "graph":
                    results = retriever.graph_search(query, limit=3)
                else:
                    # Vector search
                    query_embedding = retriever.embedding_model.encode([query])
                    scores, indices = retriever.index.search(query_embedding, k=3)
                    results = [retriever.document_metadata[i] for i in indices[0]]
                
                logger.info(f"✅ {search_type.capitalize()} search for '{query}': {len(results)} results")
                
            except Exception as e:
                logger.error(f"❌ {search_type.capitalize()} search failed for '{query}': {e}")
        
        retriever.close()
        logger.info("✅ System tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ System tests failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🚀 Starting GraphRAG Neo4j Research Framework Setup")
    logger.info("=" * 60)
    
    # Step 1: Check Neo4j connection
    if not check_neo4j_connection():
        logger.error("❌ Setup failed: Cannot connect to Neo4j")
        return False
    
    # Step 2: Initialize database
    if not initialize_database():
        logger.error("❌ Setup failed: Database initialization failed")
        return False
    
    # Step 3: Build vector index
    if not build_vector_index():
        logger.error("❌ Setup failed: Vector index build failed")
        return False
    
    # Step 4: Run system tests
    if not test_system():
        logger.error("❌ Setup failed: System tests failed")
        return False
    
    logger.info("=" * 60)
    logger.info("🎉 GraphRAG Neo4j Research Framework Setup Completed Successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the web application: streamlit run src/app/main.py")
    logger.info("2. Access Neo4j Browser: http://localhost:7474")
    logger.info("3. Run the test suite: python test_suite.py")
    logger.info("")
    logger.info("Happy researching! 🧠📊")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 