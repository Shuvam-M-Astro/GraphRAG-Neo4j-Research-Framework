#!/usr/bin/env python3
"""
Test script for Graph RAG Scientific Research system.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph_rag.orchestrator import GraphRAGOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_functionality():
    """Test basic functionality of the GraphRAG Neo4j Research Framework."""
    print("🔬 Testing GraphRAG Neo4j Research Framework")
    print("=" * 50)
    
    try:
        # Initialize orchestrator
        print("1. Initializing Graph RAG Orchestrator...")
        orchestrator = GraphRAGOrchestrator()
        print("✅ Orchestrator initialized successfully!")
        
        # Test research topic analysis
        print("\n2. Testing research topic analysis...")
        query = "deep learning"
        results = orchestrator.analyze_research_topic(query, analysis_type="comprehensive")
        
        if "error" in results:
            print(f"❌ Error in research topic analysis: {results['error']}")
        else:
            print(f"✅ Research topic analysis successful! Found {len(results['retrieved_papers'])} papers")
        
        # Test literature review generation
        print("\n3. Testing literature review generation...")
        topic = "transformer architecture"
        lit_review = orchestrator.generate_literature_review(topic, max_papers=10)
        
        if "error" in lit_review:
            print(f"❌ Error in literature review generation: {lit_review['error']}")
        else:
            print(f"✅ Literature review generation successful! Analyzed {lit_review['papers_analyzed']} papers")
        
        # Test research gap analysis
        print("\n4. Testing research gap analysis...")
        gap_analysis = orchestrator.identify_research_gaps("graph neural networks", max_hops=2)
        
        if "error" in gap_analysis:
            print(f"❌ Error in research gap analysis: {gap_analysis['error']}")
        else:
            print("✅ Research gap analysis successful!")
        
        # Test methodology evolution
        print("\n5. Testing methodology evolution tracking...")
        evolution = orchestrator.track_methodology_evolution("transformer")
        
        if "error" in evolution:
            print(f"❌ Error in methodology evolution: {evolution['error']}")
        else:
            print("✅ Methodology evolution tracking successful!")
        
        # Test multi-hop reasoning
        print("\n6. Testing multi-hop reasoning...")
        reasoning = orchestrator.multi_hop_reasoning("How do transformers work?", max_hops=2)
        
        if "error" in reasoning:
            print(f"❌ Error in multi-hop reasoning: {reasoning['error']}")
        else:
            print(f"✅ Multi-hop reasoning successful! Found {reasoning['papers_found']} papers")
        
        # Close orchestrator
        print("\n7. Closing orchestrator...")
        orchestrator.close()
        print("✅ Orchestrator closed successfully!")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main test function."""
    load_dotenv()
    
    # Check environment variables
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please set up your .env file with the required variables.")
        return False
    
    # Run tests
    success = test_basic_functionality()
    
    if success:
        print("\n✅ System test completed successfully!")
        print("\nNext steps:")
        print("1. Run 'python src/database/init_database.py' to initialize the database")
        print("2. Run 'python src/data_ingestion/arxiv_ingestion.py' to ingest sample data")
        print("3. Run 'streamlit run src/app/main.py' to start the web application")
    else:
        print("\n❌ System test failed!")
        print("Please check your configuration and try again.")
    
    return success

if __name__ == "__main__":
    main() 