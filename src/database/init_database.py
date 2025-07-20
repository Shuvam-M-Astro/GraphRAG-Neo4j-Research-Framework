"""
Database initialization script for Graph RAG Scientific Research project.
Sets up Neo4j schema and constraints for research papers, authors, methods, etc.
"""

import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jDatabase:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
                "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
                "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.method_id IS UNIQUE",
                "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.keyword_id IS UNIQUE",
                "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (j:Journal) REQUIRE j.journal_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
                "CREATE INDEX keyword_text IF NOT EXISTS FOR (k:Keyword) ON (k.text)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")

    def create_sample_data(self):
        """Create sample research data for testing."""
        with self.driver.session() as session:
            # Sample papers
            papers = [
                {
                    "paper_id": "p1",
                    "title": "Deep Learning for Natural Language Processing",
                    "abstract": "A comprehensive survey of deep learning approaches in NLP",
                    "year": 2023,
                    "journal": "Nature Machine Intelligence",
                    "citations": 150
                },
                {
                    "paper_id": "p2", 
                    "title": "Graph Neural Networks for Scientific Discovery",
                    "abstract": "Novel applications of GNNs in scientific research",
                    "year": 2023,
                    "journal": "Science",
                    "citations": 89
                },
                {
                    "paper_id": "p3",
                    "title": "Transformer Architecture in Computer Vision",
                    "abstract": "Adapting transformer models for visual tasks",
                    "year": 2022,
                    "journal": "CVPR",
                    "citations": 234
                }
            ]

            # Sample authors
            authors = [
                {"author_id": "a1", "name": "Dr. Sarah Johnson", "institution": "MIT"},
                {"author_id": "a2", "name": "Prof. Michael Chen", "institution": "Stanford"},
                {"author_id": "a3", "name": "Dr. Emily Rodriguez", "institution": "UC Berkeley"}
            ]

            # Sample methods
            methods = [
                {"method_id": "m1", "name": "Transformer", "category": "Neural Architecture"},
                {"method_id": "m2", "name": "Graph Neural Network", "category": "Graph Learning"},
                {"method_id": "m3", "name": "Attention Mechanism", "category": "Neural Component"}
            ]

            # Sample keywords
            keywords = [
                {"keyword_id": "k1", "text": "deep learning"},
                {"keyword_id": "k2", "text": "natural language processing"},
                {"keyword_id": "k3", "text": "graph neural networks"},
                {"keyword_id": "k4", "text": "computer vision"},
                {"keyword_id": "k5", "text": "transformer"}
            ]

            # Create nodes using MERGE to handle duplicates
            for paper in papers:
                session.run("""
                    MERGE (p:Paper {paper_id: $paper_id})
                    ON CREATE SET
                        p.title = $title,
                        p.abstract = $abstract,
                        p.year = $year,
                        p.journal = $journal,
                        p.citations = $citations
                """, paper)

            for author in authors:
                session.run("""
                    MERGE (a:Author {author_id: $author_id})
                    ON CREATE SET
                        a.name = $name,
                        a.institution = $institution
                """, author)

            for method in methods:
                session.run("""
                    MERGE (m:Method {method_id: $method_id})
                    ON CREATE SET
                        m.name = $name,
                        m.category = $category
                """, method)

            for keyword in keywords:
                session.run("""
                    MERGE (k:Keyword {keyword_id: $keyword_id})
                    ON CREATE SET
                        k.text = $text
                """, keyword)

            # Create relationships
            relationships = [
                # Paper-Author relationships
                ("p1", "a1", "AUTHORED_BY"),
                ("p1", "a2", "AUTHORED_BY"),
                ("p2", "a2", "AUTHORED_BY"),
                ("p2", "a3", "AUTHORED_BY"),
                ("p3", "a1", "AUTHORED_BY"),
                ("p3", "a3", "AUTHORED_BY"),
                
                # Paper-Method relationships
                ("p1", "m1", "USES_METHOD"),
                ("p1", "m3", "USES_METHOD"),
                ("p2", "m2", "USES_METHOD"),
                ("p3", "m1", "USES_METHOD"),
                ("p3", "m3", "USES_METHOD"),
                
                # Paper-Keyword relationships
                ("p1", "k1", "HAS_KEYWORD"),
                ("p1", "k2", "HAS_KEYWORD"),
                ("p2", "k1", "HAS_KEYWORD"),
                ("p2", "k3", "HAS_KEYWORD"),
                ("p3", "k1", "HAS_KEYWORD"),
                ("p3", "k4", "HAS_KEYWORD"),
                ("p3", "k5", "HAS_KEYWORD"),
                
                # Citation relationships
                ("p2", "p1", "CITES"),
                ("p3", "p1", "CITES"),
                
                # Author collaboration
                ("a1", "a2", "COLLABORATED_WITH"),
                ("a2", "a3", "COLLABORATED_WITH"),
                ("a1", "a3", "COLLABORATED_WITH")
            ]

            for source, target, relationship in relationships:
                session.run(f"""
                    MATCH (source), (target)
                    WHERE source.paper_id = $source_id OR source.author_id = $source_id
                    AND target.paper_id = $target_id OR target.author_id = $target_id
                    MERGE (source)-[r:{relationship}]->(target)
                """, {"source_id": source, "target_id": target})

            logger.info("Sample data created successfully")

def main():
    """Main function to initialize the database."""
    db = Neo4jDatabase()
    
    try:
        db.connect()
        db.create_constraints_and_indexes()
        db.create_sample_data()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main() 