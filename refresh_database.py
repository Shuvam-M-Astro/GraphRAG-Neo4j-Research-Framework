#!/usr/bin/env python3
"""
Database Refresh Script for GraphRAG Neo4j Research Framework
Clears existing data and recreates with diverse sample data.
"""

import os
import sys
import logging
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

def clear_database():
    """Clear all existing data from the database."""
    try:
        logger.info("Clearing existing database data...")
        db = Neo4jDatabase()
        db.connect()
        
        with db.driver.session() as session:
            # Delete all relationships and nodes
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("✅ Database cleared successfully")
        
        db.close()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear database: {e}")
        return False

def recreate_database():
    """Recreate the database with new sample data."""
    try:
        logger.info("Recreating database with diverse sample data...")
        db = Neo4jDatabase()
        db.connect()
        db.create_constraints_and_indexes()
        db.create_sample_data()
        db.close()
        logger.info("✅ Database recreated successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to recreate database: {e}")
        return False

def rebuild_vector_index():
    """Rebuild the vector index with new data."""
    try:
        logger.info("Rebuilding vector index with new data...")
        retriever = GraphRetriever()
        
        if retriever.documents and retriever.document_metadata:
            logger.info(f"✅ Vector index rebuilt successfully!")
            logger.info(f"   - Documents indexed: {len(retriever.documents)}")
            logger.info(f"   - Index build time: {retriever.index_build_time:.2f}s")
            
            # Test with ADHD search
            logger.info("Testing ADHD search...")
            results = retriever.hybrid_search("adhd", limit=5)
            logger.info(f"✅ ADHD search successful! Found {len(results)} results")
            
            # Show results
            for i, result in enumerate(results[:3]):
                logger.info(f"   {i+1}. {result['title']}")
            
            retriever.close()
            return True
        else:
            logger.error("❌ Vector index rebuild failed - no documents found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector index rebuild failed: {e}")
        return False

def main():
    """Main refresh function."""
    logger.info("🔄 Starting Database Refresh")
    logger.info("=" * 50)
    
    # Step 1: Clear existing data
    if not clear_database():
        logger.error("❌ Refresh failed: Cannot clear database")
        return False
    
    # Step 2: Recreate with new data
    if not recreate_database():
        logger.error("❌ Refresh failed: Cannot recreate database")
        return False
    
    # Step 3: Rebuild vector index
    if not rebuild_vector_index():
        logger.error("❌ Refresh failed: Cannot rebuild vector index")
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 Database Refresh Completed Successfully!")
    logger.info("")
    logger.info("New diverse sample data includes:")
    logger.info("• 3 Machine Learning papers")
    logger.info("• 3 ADHD research papers")
    logger.info("• 2 Psychology papers")
    logger.info("• 2 Medical research papers")
    logger.info("")
    logger.info("Now you can search for topics like:")
    logger.info("• 'adhd' - will find ADHD research papers")
    logger.info("• 'depression' - will find psychology papers")
    logger.info("• 'covid' - will find medical research")
    logger.info("• 'deep learning' - will find ML papers")
    logger.info("")
    logger.info("Try searching again in the web app!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 