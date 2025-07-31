#!/usr/bin/env python3
"""
Simple script to view Neo4j database contents
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

def view_database():
    """View the contents of the Neo4j database."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            print("🔍 Neo4j Database Contents\n")
            
            # Count total nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            total_nodes = result.single()["count"]
            print(f"📊 Total nodes: {total_nodes}")
            
            # Count nodes by type
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
            """)
            print("\n📈 Nodes by type:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # Show papers
            print("\n📚 Papers:")
            result = session.run("""
                MATCH (p:Paper)
                RETURN p.paper_id, p.title, p.year
                ORDER BY p.year DESC
            """)
            for record in result:
                print(f"  {record['p.paper_id']}: {record['p.title']} ({record['p.year']})")
            
            # Show authors
            print("\n👥 Authors:")
            result = session.run("""
                MATCH (a:Author)
                RETURN a.name, a.institution
                ORDER BY a.name
            """)
            for record in result:
                print(f"  {record['a.name']} - {record['a.institution']}")
            
            # Show methods
            print("\n🔬 Methods:")
            result = session.run("""
                MATCH (m:Method)
                RETURN m.name, m.category
                ORDER BY m.name
            """)
            for record in result:
                print(f"  {record['m.name']} ({record['m.category']})")
            
            # Show keywords
            print("\n🏷️ Keywords:")
            result = session.run("""
                MATCH (k:Keyword)
                RETURN k.text
                ORDER BY k.text
            """)
            for record in result:
                print(f"  {record['k.text']}")
            
            # Count relationships
            print("\n🔗 Relationships:")
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # Show some relationships
            print("\n🔗 Sample Relationships:")
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN labels(a)[0] as from_type, type(r) as rel_type, labels(b)[0] as to_type, count(r) as count
                ORDER BY count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"  {record['from_type']} -[{record['rel_type']}]-> {record['to_type']}: {record['count']}")
                
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    view_database() 