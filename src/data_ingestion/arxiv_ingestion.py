"""
ArXiv Data Ingestion for Graph RAG Scientific Research
Fetches and processes ArXiv papers for the knowledge graph.
"""

import os
import logging
import arxiv
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
import time
import json

load_dotenv()

logger = logging.getLogger(__name__)

class ArXivIngestion:
    def __init__(self):
        """Initialize ArXiv ingestion with Neo4j connection."""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        self.connect()

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

    def search_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search for papers on ArXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            
        Returns:
            List of paper dictionaries
        """
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in search.results():
                paper = {
                    "paper_id": result.entry_id.split('/')[-1],
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [author.name for author in result.authors],
                    "published_date": result.published.date(),
                    "updated_date": result.updated.date(),
                    "categories": result.categories,
                    "pdf_url": result.pdf_url,
                    "journal_ref": result.journal_ref,
                    "doi": result.doi
                }
                papers.append(paper)
                
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []

    def extract_keywords(self, abstract: str, title: str) -> List[str]:
        """
        Extract keywords from abstract and title.
        
        Args:
            abstract: Paper abstract
            title: Paper title
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction (can be enhanced with NLP)
        text = f"{title} {abstract}".lower()
        
        # Common research keywords
        keywords = [
            "deep learning", "machine learning", "neural networks", "transformer",
            "graph neural networks", "computer vision", "natural language processing",
            "reinforcement learning", "optimization", "algorithm", "framework",
            "architecture", "model", "training", "inference", "evaluation",
            "benchmark", "dataset", "performance", "accuracy", "efficiency"
        ]
        
        extracted_keywords = []
        for keyword in keywords:
            if keyword in text:
                extracted_keywords.append(keyword)
        
        return extracted_keywords

    def extract_methodologies(self, abstract: str, title: str) -> List[str]:
        """
        Extract methodologies from abstract and title.
        
        Args:
            abstract: Paper abstract
            title: Paper title
            
        Returns:
            List of extracted methodologies
        """
        # Simple methodology extraction (can be enhanced with NLP)
        text = f"{title} {abstract}".lower()
        
        # Common methodologies
        methodologies = [
            "transformer", "attention mechanism", "graph neural network",
            "convolutional neural network", "recurrent neural network",
            "long short-term memory", "lstm", "gru", "cnn", "rnn",
            "bert", "gpt", "resnet", "vgg", "inception", "yolo",
            "faster r-cnn", "mask r-cnn", "u-net", "gan", "vae"
        ]
        
        extracted_methods = []
        for method in methodologies:
            if method in text:
                extracted_methods.append(method)
        
        return extracted_methods

    def store_paper(self, paper: Dict) -> bool:
        """
        Store a paper in Neo4j database.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Create paper node
                session.run("""
                    MERGE (p:Paper {paper_id: $paper_id})
                    SET p.title = $title,
                        p.abstract = $abstract,
                        p.published_date = $published_date,
                        p.updated_date = $updated_date,
                        p.pdf_url = $pdf_url,
                        p.journal_ref = $journal_ref,
                        p.doi = $doi,
                        p.year = $year
                """, {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "published_date": str(paper["published_date"]),
                    "updated_date": str(paper["updated_date"]),
                    "pdf_url": paper["pdf_url"],
                    "journal_ref": paper["journal_ref"] or "",
                    "doi": paper["doi"] or "",
                    "year": paper["published_date"].year
                })
                
                # Create author nodes and relationships
                for author_name in paper["authors"]:
                    session.run("""
                        MERGE (a:Author {name: $author_name})
                        MERGE (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:AUTHORED_BY]->(a)
                    """, {
                        "author_name": author_name,
                        "paper_id": paper["paper_id"]
                    })
                
                # Create keyword nodes and relationships
                keywords = self.extract_keywords(paper["abstract"], paper["title"])
                for keyword in keywords:
                    session.run("""
                        MERGE (k:Keyword {text: $keyword})
                        MERGE (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:HAS_KEYWORD]->(k)
                    """, {
                        "keyword": keyword,
                        "paper_id": paper["paper_id"]
                    })
                
                # Create methodology nodes and relationships
                methods = self.extract_methodologies(paper["abstract"], paper["title"])
                for method in methods:
                    session.run("""
                        MERGE (m:Method {name: $method})
                        MERGE (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:USES_METHOD]->(m)
                    """, {
                        "method": method,
                        "paper_id": paper["paper_id"]
                    })
                
                # Create category nodes and relationships
                for category in paper["categories"]:
                    session.run("""
                        MERGE (c:Category {name: $category})
                        MERGE (p:Paper {paper_id: $paper_id})
                        MERGE (p)-[:BELONGS_TO]->(c)
                    """, {
                        "category": category,
                        "paper_id": paper["paper_id"]
                    })
                
                logger.info(f"Successfully stored paper: {paper['paper_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing paper {paper['paper_id']}: {e}")
            return False

    def ingest_papers(self, queries: List[str], max_results_per_query: int = 50) -> Dict[str, Any]:
        """
        Ingest papers from ArXiv for multiple queries.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            Ingestion statistics
        """
        total_papers = 0
        successful_stores = 0
        failed_stores = 0
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            # Search for papers
            papers = self.search_papers(query, max_results_per_query)
            total_papers += len(papers)
            
            # Store papers
            for paper in papers:
                if self.store_paper(paper):
                    successful_stores += 1
                else:
                    failed_stores += 1
                
                # Rate limiting
                time.sleep(0.1)
        
        stats = {
            "total_papers_found": total_papers,
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "success_rate": successful_stores / total_papers if total_papers > 0 else 0
        }
        
        logger.info(f"Ingestion completed. Stats: {stats}")
        return stats

    def create_collaboration_relationships(self):
        """
        Create collaboration relationships between authors.
        """
        try:
            with self.driver.session() as session:
                # Find authors who have collaborated on papers
                session.run("""
                    MATCH (a1:Author)-[:AUTHORED_BY]->(p:Paper)<-[:AUTHORED_BY]-(a2:Author)
                    WHERE a1.name < a2.name
                    MERGE (a1)-[:COLLABORATED_WITH]->(a2)
                """)
                
                logger.info("Created collaboration relationships")
                
        except Exception as e:
            logger.error(f"Error creating collaboration relationships: {e}")

    def create_citation_relationships(self):
        """
        Create citation relationships between papers based on references.
        Note: This is a simplified approach. Real citation data would require
        parsing PDFs or using external citation databases.
        """
        try:
            with self.driver.session() as session:
                # Create citations based on keyword similarity (simplified)
                session.run("""
                    MATCH (p1:Paper)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Paper)
                    WHERE p1.paper_id <> p2.paper_id
                    AND p1.year >= p2.year
                    MERGE (p1)-[:CITES]->(p2)
                """)
                
                logger.info("Created citation relationships")
                
        except Exception as e:
            logger.error(f"Error creating citation relationships: {e}")

def main():
    """Main function for ArXiv ingestion."""
    # Example queries for scientific research
    queries = [
        "deep learning",
        "graph neural networks",
        "transformer architecture",
        "computer vision",
        "natural language processing",
        "machine learning",
        "neural networks",
        "attention mechanism",
        "reinforcement learning",
        "optimization algorithms"
    ]
    
    ingestion = ArXivIngestion()
    
    try:
        # Ingest papers
        stats = ingestion.ingest_papers(queries, max_results_per_query=30)
        print(f"Ingestion stats: {stats}")
        
        # Create relationships
        ingestion.create_collaboration_relationships()
        ingestion.create_citation_relationships()
        
        print("ArXiv ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"ArXiv ingestion failed: {e}")
    finally:
        ingestion.close()

if __name__ == "__main__":
    main() 