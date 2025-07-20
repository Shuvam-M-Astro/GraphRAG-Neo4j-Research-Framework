#!/usr/bin/env python3
"""
Clean test without any APOC dependencies
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

load_dotenv()

def test_clean_neo4j():
    """Test Neo4j with only basic operations"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Test 1: Basic count
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"✅ Total nodes: {count}")
            
            # Test 2: Get papers
            result = session.run("MATCH (p:Paper) RETURN p.paper_id as id, p.title as title LIMIT 3")
            papers = [record for record in result]
            print(f"✅ Found {len(papers)} papers")
            
            # Test 3: Simple relationship query (no APOC)
            if papers:
                paper_ids = [p["id"] for p in papers]
                print(f"✅ Testing with paper IDs: {paper_ids}")
                
                # Simple query without APOC
                result = session.run("""
                    MATCH (p:Paper)
                    WHERE p.paper_id IN $paper_ids
                    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                    RETURN p.paper_id as paper_id, p.title as title, 
                           collect(DISTINCT k.text) as keywords
                """, {"paper_ids": paper_ids})
                
                results = [record for record in result]
                print(f"✅ Found {len(results)} paper-keyword relationships")
        
        driver.close()
        print("✅ All clean Neo4j tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Clean Neo4j test failed: {e}")
        return False

def test_simple_search():
    """Test simple search without complex graph operations"""
    try:
        from graph_rag.graph_retriever import GraphRetriever
        
        retriever = GraphRetriever()
        
        # Test simple vector search only
        query_embedding = retriever.embedding_model.encode(["quantum computing"])
        scores, indices = retriever.index.search(query_embedding, k=min(5, len(retriever.documents)))
        
        print(f"✅ Vector search returned {len(indices[0])} results")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Simple search test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧹 Clean APOC-Free Test")
    print("=" * 30)
    
    print("\n1. Testing clean Neo4j operations...")
    neo4j_ok = test_clean_neo4j()
    
    if neo4j_ok:
        print("\n2. Testing simple search...")
        search_ok = test_simple_search()
        
        if search_ok:
            print("\n✅ All clean tests passed! No APOC procedures used.")
        else:
            print("\n❌ Simple search test failed.")
    else:
        print("\n❌ Clean Neo4j test failed.") 