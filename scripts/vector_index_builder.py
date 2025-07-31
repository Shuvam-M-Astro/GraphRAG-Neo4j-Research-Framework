#!/usr/bin/env python3
"""
Vector Index Builder for GraphRAG Neo4j Research Framework
Builds the FAISS vector index from documents in Neo4j database.
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

from graph_rag.graph_retriever import GraphRetriever

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Build the vector index for the GraphRAG system."""
    try:
        logger.info("Starting vector index build...")
        
        # Initialize GraphRetriever (this will automatically build the vector index)
        logger.info("Initializing GraphRetriever...")
        retriever = GraphRetriever()
        
        # Verify the index was built successfully
        if retriever.documents and retriever.document_metadata:
            logger.info(f"✅ Vector index built successfully!")
            logger.info(f"   - Documents indexed: {len(retriever.documents)}")
            logger.info(f"   - Index build time: {retriever.index_build_time:.2f}s")
            logger.info(f"   - Vector dimension: {retriever.vector_dim}")
            
            # Test the index with a simple search
            logger.info("Testing vector index with sample query...")
            try:
                results = retriever.hybrid_search("deep learning", limit=3)
                logger.info(f"✅ Test search successful! Found {len(results)} results")
                
                # Show first result
                if results:
                    first_result = results[0]
                    logger.info(f"   Sample result: {first_result['title']}")
                    
            except Exception as e:
                logger.error(f"❌ Test search failed: {e}")
                return False
                
        else:
            logger.error("❌ Vector index build failed - no documents found")
            return False
            
        logger.info("🎉 Vector index build completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector index build failed: {e}")
        return False
    finally:
        # Clean up
        if 'retriever' in locals():
            retriever.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 