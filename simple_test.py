#!/usr/bin/env python3
"""
Simple test to isolate APOC error
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

load_dotenv()

def test_simple_neo4j():
    """Test simple Neo4j operations"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password123")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Test 1: Simple count
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"Total nodes: {count}")
            
            # Test 2: Get papers
            result = session.run("MATCH (p:Paper) RETURN p.paper_id as id, p.title as title LIMIT 5")
            papers = [record for record in result]
            print(f"Found {len(papers)} papers")
            
            # Test 3: Test the exact query that might be causing issues
            if papers:
                paper_ids = [p["id"] for p in papers]
                print(f"Testing with paper IDs: {paper_ids}")
                
                # This is the query that was causing APOC issues
                result = session.run("""
                    MATCH (start:Paper)
                    WHERE start.paper_id IN $paper_ids
                    WITH start
                    OPTIONAL MATCH (start)-[:CITES|AUTHORED_BY|USES_METHOD|HAS_KEYWORD|COLLABORATED_WITH*1..2]-(related)
                    RETURN DISTINCT related as node
                    LIMIT 10
                """, {"paper_ids": paper_ids})
                
                related_nodes = [record for record in result]
                print(f"Found {len(related_nodes)} related nodes")
        
        driver.close()
        print("✅ All Neo4j tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Neo4j test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple APOC Test")
    print("=" * 30)
    test_simple_neo4j() 