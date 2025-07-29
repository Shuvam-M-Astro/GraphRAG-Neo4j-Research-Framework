"""
ArXiv Web Scraper for Graph RAG Scientific Research
Scrapes actual data from arXiv search results pages.
"""

import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
import time
import re
from urllib.parse import urljoin, urlparse
from datetime import datetime
import json

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
from custom_types import (
    PaperId, PaperMetadata, ValidationError, DatabaseError, 
    is_paper_metadata, to_paper_metadata
)

load_dotenv()

logger = logging.getLogger(__name__)

class ArXivWebScraper:
    def __init__(self):
        """
        Initialize ArXiv web scraper with Neo4j connection.
        
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
            
            # Web scraping configuration
            self.base_url = "https://arxiv.org"
            self.search_url = "https://arxiv.org/search/"
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Performance tracking
            self.scraping_stats = {
                "total_papers_found": 0,
                "successful_stores": 0,
                "failed_stores": 0,
                "total_processing_time": 0.0,
                "pages_scraped": 0
            }
            
            self.connect()
            
        except Exception as e:
            logger.error(f"ArXivWebScraper initialization failed: {e}")
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

    def search_papers_web(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Search for papers on ArXiv using web scraping.
        
        Args:
            query: Search query
            max_results: Maximum number of results to fetch
            
        Returns:
            List of paper dictionaries
        """
        start_time = time.time()
        papers = []
        page = 0
        results_per_page = 50
        
        try:
            while len(papers) < max_results:
                # Construct search URL
                search_params = {
                    'query': query,
                    'searchtype': 'all',
                    'source': 'header',
                    'start': page * results_per_page,
                    'max_results': min(results_per_page, max_results - len(papers))
                }
                
                url = f"{self.search_url}?{'&'.join([f'{k}={v}' for k, v in search_params.items()])}"
                logger.info(f"Scraping page {page + 1} from: {url}")
                
                # Make request
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract paper entries
                paper_entries = soup.find_all('li', class_='arxiv-result')
                if not paper_entries:
                    # Try alternative selectors
                    paper_entries = soup.find_all('div', class_='result')
                
                if not paper_entries:
                    logger.warning(f"No paper entries found on page {page + 1}")
                    break
                
                page_papers = []
                for entry in paper_entries:
                    try:
                        paper = self._extract_paper_from_entry(entry)
                        if paper:
                            page_papers.append(paper)
                    except Exception as e:
                        logger.warning(f"Error extracting paper from entry: {e}")
                        continue
                
                papers.extend(page_papers)
                self.scraping_stats["pages_scraped"] += 1
                
                # Rate limiting - be more respectful to ArXiv
                time.sleep(2)  # Increased delay to be more respectful
                
                # Check if we've reached the end
                if len(page_papers) < results_per_page:
                    break
                
                page += 1
                
                # Safety check
                if page > 10:  # Max 10 pages
                    break
            
            processing_time = time.time() - start_time
            self.scraping_stats["total_papers_found"] = len(papers)
            self.scraping_stats["total_processing_time"] += processing_time
            
            logger.info(f"Found {len(papers)} papers for query '{query}' in {processing_time:.2f}s")
            return papers
            
        except Exception as e:
            logger.error(f"Error scraping ArXiv: {e}")
            return []

    def _extract_paper_from_entry(self, entry) -> Optional[Dict[str, Any]]:
        """
        Extract paper information from a search result entry.
        
        Args:
            entry: BeautifulSoup element representing a paper entry
            
        Returns:
            Paper dictionary or None if extraction fails
        """
        try:
            # Extract arXiv ID
            arxiv_id_elem = entry.find('a', href=re.compile(r'/abs/\d+\.\d+'))
            if not arxiv_id_elem:
                return None
            
            arxiv_id = arxiv_id_elem['href'].split('/')[-1]
            
            # Extract title
            title_elem = entry.find('p', class_='title') or entry.find('h1') or entry.find('a', href=re.compile(r'/abs/'))
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Extract authors
            authors_elem = entry.find('p', class_='authors') or entry.find('div', class_='authors')
            authors = []
            if authors_elem:
                author_links = authors_elem.find_all('a')
                authors = [link.get_text(strip=True) for link in author_links]
            
            # Extract abstract
            abstract_elem = entry.find('p', class_='abstract') or entry.find('span', class_='abstract')
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""
            
            # Extract categories
            categories = []
            category_elems = entry.find_all('span', class_='primary-subject') or entry.find_all('span', class_='category')
            for cat_elem in category_elems:
                cat_text = cat_elem.get_text(strip=True)
                if cat_text:
                    categories.append(cat_text)
            
            # Extract submission date
            date_elem = entry.find('span', class_='date') or entry.find('div', class_='date')
            published_date = None
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                try:
                    # Parse date (format: "Submitted 17 April, 2025; originally announced April 2025.")
                    date_match = re.search(r'Submitted (\d+ \w+, \d+)', date_text)
                    if date_match:
                        date_str = date_match.group(1)
                        published_date = datetime.strptime(date_str, "%d %B, %Y").date()
                except:
                    pass
            
            # Extract PDF URL
            pdf_url = ""
            pdf_elem = entry.find('a', href=re.compile(r'\.pdf$'))
            if pdf_elem:
                pdf_url = urljoin(self.base_url, pdf_elem['href'])
            
            # Extract DOI
            doi = ""
            doi_elem = entry.find('a', href=re.compile(r'doi\.org'))
            if doi_elem:
                doi = doi_elem['href']
            
            # Basic validation
            if not title or not abstract:
                return None
            
            if len(abstract) < 10:
                return None
            
            paper = {
                "paper_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published_date": published_date or datetime.now().date(),
                "updated_date": published_date or datetime.now().date(),
                "categories": categories,
                "pdf_url": pdf_url,
                "journal_ref": "",
                "doi": doi
            }
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error extracting paper data: {e}")
            return None

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
            "benchmark", "dataset", "performance", "accuracy", "efficiency",
            "adhd", "attention deficit", "hyperactivity", "neurobiology",
            "cognitive", "behavioral", "therapy", "treatment", "diagnosis",
            "depression", "anxiety", "mental health", "psychology", "neuroscience",
            "covid", "cancer", "medical", "clinical", "pharmacology"
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
            "faster r-cnn", "mask r-cnn", "u-net", "gan", "vae",
            "cognitive behavioral therapy", "meta-analysis", "randomized controlled trial",
            "fmri", "eeg", "neuroimaging", "longitudinal study", "systematic review"
        ]
        
        extracted_methods = []
        for method in methodologies:
            if method in text:
                extracted_methods.append(method)
        
        return extracted_methods

    def store_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Store a paper in Neo4j database.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            True if successful, False otherwise
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
                self.scraping_stats["successful_stores"] += 1
                self.scraping_stats["total_processing_time"] += processing_time
                
                logger.info(f"Successfully stored paper {validated_paper.paper_id} in {processing_time:.2f}s")
                return True
                
        except Exception as e:
            self.scraping_stats["failed_stores"] += 1
            logger.error(f"Error storing paper {paper.get('paper_id', 'unknown')}: {e}")
            return False

    def scrape_and_store_papers(self, queries: List[str], max_results_per_query: int = 50) -> Dict[str, Any]:
        """
        Scrape and store papers from ArXiv for multiple queries.
        
        Args:
            queries: List of search queries
            max_results_per_query: Maximum results per query
            
        Returns:
            Scraping statistics
        """
        total_papers = 0
        successful_stores = 0
        failed_stores = 0
        
        for query in queries:
            logger.info(f"Processing query: {query}")
            
            # Search for papers
            papers = self.search_papers_web(query, max_results_per_query)
            total_papers += len(papers)
            
            # Store papers
            for paper in papers:
                if self.store_paper(paper):
                    successful_stores += 1
                else:
                    failed_stores += 1
                
                # Rate limiting - be more respectful to ArXiv
                time.sleep(1)  # Increased delay to be more respectful
        
        stats = {
            "total_papers_found": total_papers,
            "successful_stores": successful_stores,
            "failed_stores": failed_stores,
            "success_rate": successful_stores / total_papers if total_papers > 0 else 0,
            "pages_scraped": self.scraping_stats["pages_scraped"],
            "total_processing_time": self.scraping_stats["total_processing_time"]
        }
        
        logger.info(f"Scraping completed. Stats: {stats}")
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
    """Main function for ArXiv web scraping."""
    # Example queries for scientific research
    queries = [
        "adhd",
        "attention deficit hyperactivity disorder",
        "deep learning",
        "graph neural networks",
        "transformer architecture",
        "computer vision",
        "natural language processing",
        "machine learning",
        "neural networks",
        "attention mechanism"
    ]
    
    scraper = ArXivWebScraper()
    
    try:
        # Scrape and store papers
        stats = scraper.scrape_and_store_papers(queries, max_results_per_query=30)
        print(f"Scraping stats: {stats}")
        
        # Create relationships
        scraper.create_collaboration_relationships()
        scraper.create_citation_relationships()
        
        print("ArXiv web scraping completed successfully!")
        
    except Exception as e:
        logger.error(f"ArXiv web scraping failed: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 