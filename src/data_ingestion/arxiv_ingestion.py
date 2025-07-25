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
import re
from functools import wraps

# Import validation utilities
import sys
import os
# Add the src directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from validation import (
    validate_database_config, validate_paper_data, Paper, DatabaseConfig
)
# Import from our custom types module
from custom_types import (
    PaperId, PaperMetadata, ValidationError, DatabaseError, 
    is_paper_metadata, to_paper_metadata
)

load_dotenv()

logger = logging.getLogger(__name__)

def validate_ingestion_inputs(func):
    """Decorator to validate ArXivIngestion method inputs."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Validate query parameter if present
            if 'query' in kwargs and kwargs['query']:
                if not isinstance(kwargs['query'], str) or not kwargs['query'].strip():
                    raise ValidationError("Query must be a non-empty string")
            
            # Validate max_results parameter
            if 'max_results' in kwargs:
                value = kwargs['max_results']
                if not isinstance(value, int) or value <= 0 or value > 1000:
                    raise ValidationError("max_results must be a positive integer <= 1000")
            
            # Validate paper data if present
            if 'paper' in kwargs and kwargs['paper']:
                if not isinstance(kwargs['paper'], dict):
                    raise ValidationError("Paper must be a dictionary")
            
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Input validation failed for {func.__name__}: {e}")
            raise ValidationError(f"Input validation failed: {e}")
    return wrapper

class ArXivIngestion:
    def __init__(self):
        """
        Initialize ArXiv ingestion with Neo4j connection.
        
        Raises:
            ValidationError: If configuration is invalid
            DatabaseError: If database connection fails
        """
        try:
            # Validate database configuration
            db_config = validate_database_config(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password")
            )
            
            self.uri = db_config.uri
            self.user = db_config.user
            self.password = db_config.password
            self.driver = None
            
            # Performance tracking
            self.ingestion_stats = {
                "total_papers_processed": 0,
                "successful_stores": 0,
                "failed_stores": 0,
                "total_processing_time": 0.0
            }
            
            self.connect()
            
        except Exception as e:
            logger.error(f"ArXivIngestion initialization failed: {e}")
            raise

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

    @validate_ingestion_inputs
    @type_checked
    def search_papers(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch (1-1000)
            
        Returns:
            List of paper dictionaries
            
        Raises:
            ValidationError: If input parameters are invalid
        """
        start_time = time.time()
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            papers = []
            for result in search.results():
                try:
                    # Validate and clean paper data
                    paper = {
                        "paper_id": result.entry_id.split('/')[-1],
                        "title": result.title.strip() if result.title else "",
                        "abstract": result.summary.strip() if result.summary else "",
                        "authors": [author.name.strip() for author in result.authors if author.name.strip()],
                        "published_date": result.published.date(),
                        "updated_date": result.updated.date(),
                        "categories": result.categories,
                        "pdf_url": result.pdf_url,
                        "journal_ref": result.journal_ref,
                        "doi": result.doi
                    }
                    
                    # Basic validation
                    if not paper["title"] or not paper["abstract"]:
                        logger.warning(f"Skipping paper with missing title or abstract: {paper['paper_id']}")
                        continue
                    
                    if len(paper["abstract"]) < 10:
                        logger.warning(f"Skipping paper with very short abstract: {paper['paper_id']}")
                        continue
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.warning(f"Error processing ArXiv result: {e}")
                    continue
                
            processing_time = time.time() - start_time
            logger.info(f"Found {len(papers)} valid papers for query '{query}' in {processing_time:.2f}s")
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

    @validate_ingestion_inputs
    @type_checked
    def store_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Store a paper in Neo4j database.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValidationError: If paper data is invalid
            DatabaseError: If database operation fails
        """
        start_time = time.time()
        
        try:
            # Validate paper data
            if not isinstance(paper, dict):
                raise ValidationError("Paper must be a dictionary")
            
            required_fields = ["paper_id", "title", "abstract", "published_date"]
            for field in required_fields:
                if field not in paper or not paper[field]:
                    raise ValidationError(f"Missing required field: {field}")
            
            # Validate paper with Pydantic model
            try:
                paper_data = {
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "year": paper["published_date"].year,
                    "journal": paper.get("journal_ref"),
                    "citations": 0,  # ArXiv papers start with 0 citations
                    "authors": paper.get("authors", []),
                    "keywords": [],
                    "methods": []
                }
                
                validated_paper = Paper(**paper_data)
                
            except Exception as e:
                logger.error(f"Paper validation failed: {e}")
                raise ValidationError(f"Invalid paper data: {e}")
            
            with self.driver.session() as session:
                # Create paper node with validation
                session.run("""
                    MERGE (p:Paper {paper_id: $paper_id})
                    SET p.title = $title,
                        p.abstract = $abstract,
                        p.published_date = $published_date,
                        p.updated_date = $updated_date,
                        p.pdf_url = $pdf_url,
                        p.journal_ref = $journal_ref,
                        p.doi = $doi,
                        p.year = $year,
                        p.citations = $citations
                """, {
                    "paper_id": validated_paper.paper_id,
                    "title": validated_paper.title,
                    "abstract": validated_paper.abstract,
                    "published_date": str(paper["published_date"]),
                    "updated_date": str(paper["updated_date"]),
                    "pdf_url": paper.get("pdf_url", ""),
                    "journal_ref": paper.get("journal_ref", ""),
                    "doi": paper.get("doi", ""),
                    "year": validated_paper.year,
                    "citations": validated_paper.citations
                })
                
                # Create author nodes and relationships
                for author_name in validated_paper.authors:
                    if author_name.strip():
                        session.run("""
                            MERGE (a:Author {name: $author_name})
                            MERGE (p:Paper {paper_id: $paper_id})
                            MERGE (p)-[:AUTHORED_BY]->(a)
                        """, {
                            "author_name": author_name.strip(),
                            "paper_id": validated_paper.paper_id
                        })
                
                # Create keyword nodes and relationships
                keywords = self.extract_keywords(validated_paper.abstract, validated_paper.title)
                for keyword in keywords:
                    if keyword.strip():
                        session.run("""
                            MERGE (k:Keyword {text: $keyword})
                            MERGE (p:Paper {paper_id: $paper_id})
                            MERGE (p)-[:HAS_KEYWORD]->(k)
                        """, {
                            "keyword": keyword.strip().lower(),
                            "paper_id": validated_paper.paper_id
                        })
                
                # Create method nodes and relationships
                methods = self.extract_methodologies(validated_paper.abstract, validated_paper.title)
                for method in methods:
                    if method.strip():
                        session.run("""
                            MERGE (m:Method {name: $method})
                            MERGE (p:Paper {paper_id: $paper_id})
                            MERGE (p)-[:USES_METHOD]->(m)
                        """, {
                            "method": method.strip(),
                            "paper_id": validated_paper.paper_id
                        })
                
                # Create category nodes and relationships
                for category in paper.get("categories", []):
                    if category.strip():
                        session.run("""
                            MERGE (c:Category {name: $category})
                            MERGE (p:Paper {paper_id: $paper_id})
                            MERGE (p)-[:BELONGS_TO]->(c)
                        """, {
                            "category": category.strip(),
                            "paper_id": validated_paper.paper_id
                        })
                
                # Update statistics
                processing_time = time.time() - start_time
                self.ingestion_stats["successful_stores"] += 1
                self.ingestion_stats["total_processing_time"] += processing_time
                
                logger.info(f"Successfully stored paper {validated_paper.paper_id} in {processing_time:.2f}s")
                return True
                
        except Exception as e:
            self.ingestion_stats["failed_stores"] += 1
            logger.error(f"Error storing paper {paper.get('paper_id', 'unknown')}: {e}")
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