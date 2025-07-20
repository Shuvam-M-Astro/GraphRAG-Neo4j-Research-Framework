#!/usr/bin/env python3
"""
Debug script to identify APOC procedure calls
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

load_dotenv()

def test_neo4j_connection():
    """Test basic Neo4j connection"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ Neo4j connection successful")
        
        # Test basic query
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✅ Database has {count} nodes")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def test_graph_retriever():
    """Test GraphRetriever without APOC"""
    try:
        from graph_rag.graph_retriever import GraphRetriever
        
        retriever = GraphRetriever()
        print("✅ GraphRetriever initialized successfully")
        
        # Test simple search
        results = retriever.hybrid_search("quantum computing", limit=5)
        print(f"✅ Hybrid search returned {len(results)} results")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ GraphRetriever failed: {e}")
        return False

def test_orchestrator():
    """Test GraphRAGOrchestrator"""
    try:
        from graph_rag.orchestrator import GraphRAGOrchestrator
        
        orchestrator = GraphRAGOrchestrator()
        print("✅ GraphRAGOrchestrator initialized successfully")
        
        # Test research topic analysis
        results = orchestrator.analyze_research_topic("quantum computing", "comprehensive")
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False
        else:
            print(f"✅ Analysis successful, found {len(results['retrieved_papers'])} papers")
            return True
            
    except Exception as e:
        print(f"❌ GraphRAGOrchestrator failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Debugging APOC procedure calls...")
    print("=" * 50)
    
    # Test 1: Basic Neo4j connection
    print("\n1. Testing Neo4j connection...")
    neo4j_ok = test_neo4j_connection()
    
    if neo4j_ok:
        # Test 2: GraphRetriever
        print("\n2. Testing GraphRetriever...")
        retriever_ok = test_graph_retriever()
        
        if retriever_ok:
            # Test 3: Orchestrator
            print("\n3. Testing GraphRAGOrchestrator...")
            orchestrator_ok = test_orchestrator()
            
            if orchestrator_ok:
                print("\n✅ All tests passed! No APOC procedures found.")
            else:
                print("\n❌ Orchestrator test failed - this is likely where the APOC error occurs.")
        else:
            print("\n❌ GraphRetriever test failed.")
    else:
        print("\n❌ Neo4j connection failed - check your database setup.") 