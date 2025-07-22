"""
Graph RAG Retriever for Scientific Research
Combines graph traversal with vector similarity search for multi-hop reasoning.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import time
from functools import wraps

# Import validation utilities
from ..validation import (
    validate_search_params, validate_database_config, validate_model_config,
    validate_paper_data, Paper, SearchQuery, DatabaseConfig, ModelConfig,
    type_checked, validate_inputs
)
from ..types import (
    PaperId, PaperMetadata, PaperWithScore, SearchScore, SearchResult,
    EmbeddingVector, EmbeddingMatrix, Neo4jResult, ValidationError,
    DatabaseError, SearchError, ModelError, is_paper_metadata
)

load_dotenv()

logger = logging.getLogger(__name__)

def validate_retriever_inputs(func):
    """Decorator to validate GraphRetriever method inputs."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # Validate query parameter if present
            if 'query' in kwargs and kwargs['query']:
                if not isinstance(kwargs['query'], str) or not kwargs['query'].strip():
                    raise ValidationError("Query must be a non-empty string")
            
            # Validate numeric parameters
            for param in ['max_hops', 'limit', 'max_results']:
                if param in kwargs:
                    value = kwargs[param]
                    if not isinstance(value, int) or value <= 0:
                        raise ValidationError(f"{param} must be a positive integer")
            
            # Validate alpha parameter for hybrid search
            if 'alpha' in kwargs:
                alpha = kwargs['alpha']
                if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
                    raise ValidationError("Alpha must be a number between 0 and 1")
            
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Input validation failed for {func.__name__}: {e}")
            raise ValidationError(f"Input validation failed: {e}")
    return wrapper

class GraphRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Graph RAG retriever.
        
        Args:
            model_name: Sentence transformer model for embeddings
            
        Raises:
            ValidationError: If model configuration is invalid
            DatabaseError: If database connection fails
            ModelError: If embedding model initialization fails
        """
        try:
            # Validate model configuration
            model_config = validate_model_config(embedding_model=model_name)
            
            # Validate database configuration
            db_config = validate_database_config(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USER", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password")
            )
            
            self.uri = db_config.uri
            self.user = db_config.user
            self.password = db_config.password
            self.model_name = model_config.embedding_model
            self.driver = None
            
            # Initialize sentence transformer for embeddings
            try:
                self.embedding_model = SentenceTransformer(self.model_name)
                self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Initialized embedding model: {self.model_name} (dim: {self.vector_dim})")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                raise ModelError(f"Embedding model initialization failed: {e}")
            
            # Initialize FAISS index for vector search
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.documents: List[str] = []
            self.document_metadata: List[PaperMetadata] = []
            
            # Performance tracking
            self.search_times: List[float] = []
            self.index_build_time: Optional[float] = None
            
            self.connect()
            self.build_vector_index()
            
        except Exception as e:
            logger.error(f"GraphRetriever initialization failed: {e}")
            raise

    def connect(self) -> None:
        """
        Establish connection to Neo4j database.
        
        Raises:
            DatabaseError: If connection fails
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1 as test")
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise DatabaseError(f"Database connection failed: {e}")

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def build_vector_index(self) -> None:
        """
        Build FAISS index from documents in Neo4j.
        
        Raises:
            DatabaseError: If database query fails
            ModelError: If embedding generation fails
        """
        start_time = time.time()
        
        try:
            with self.driver.session() as session:
                # Get all papers with their metadata
                result = session.run("""
                    MATCH (p:Paper)
                    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                    OPTIONAL MATCH (p)-[:AUTHORED_BY]->(a:Author)
                    OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
                    RETURN p, 
                           collect(DISTINCT k.text) as keywords,
                           collect(DISTINCT a.name) as authors,
                           collect(DISTINCT m.name) as methods
                """)
                
                documents: List[str] = []
                metadata: List[PaperMetadata] = []
                
                for record in result:
                    paper = record["p"]
                    keywords = record["keywords"]
                    authors = record["authors"]
                    methods = record["methods"]
                    
                    # Validate paper data
                    try:
                        paper_data = {
                            "paper_id": paper["paper_id"],
                            "title": paper["title"],
                            "abstract": paper["abstract"],
                            "year": paper["year"],
                            "journal": paper.get("journal"),
                            "citations": paper.get("citations", 0),
                            "keywords": keywords if keywords else [],
                            "authors": authors if authors else [],
                            "methods": methods if methods else []
                        }
                        
                        # Validate with Pydantic model
                        validated_paper = Paper(**paper_data)
                        
                        # Create document text
                        doc_text = f"Title: {validated_paper.title}\n"
                        doc_text += f"Abstract: {validated_paper.abstract}\n"
                        if validated_paper.keywords:
                            doc_text += f"Keywords: {', '.join(validated_paper.keywords)}\n"
                        if validated_paper.authors:
                            doc_text += f"Authors: {', '.join(validated_paper.authors)}\n"
                        if validated_paper.methods:
                            doc_text += f"Methods: {', '.join(validated_paper.methods)}\n"
                        
                        documents.append(doc_text)
                        metadata.append(PaperMetadata(
                            paper_id=validated_paper.paper_id,
                            title=validated_paper.title,
                            abstract=validated_paper.abstract,
                            year=validated_paper.year,
                            journal=validated_paper.journal,
                            citations=validated_paper.citations,
                            keywords=validated_paper.keywords,
                            authors=validated_paper.authors,
                            methods=validated_paper.methods
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Skipping invalid paper {paper.get('paper_id', 'unknown')}: {e}")
                        continue
                
                # Create embeddings and add to FAISS index
                if documents:
                    try:
                        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
                        self.index.add(embeddings.astype('float32'))
                        self.documents = documents
                        self.document_metadata = metadata
                        
                        self.index_build_time = time.time() - start_time
                        logger.info(f"Built vector index with {len(documents)} documents in {self.index_build_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Failed to create embeddings: {e}")
                        raise ModelError(f"Embedding generation failed: {e}")
                else:
                    logger.warning("No valid documents found for vector index")
                    
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            raise DatabaseError(f"Vector index build failed: {e}")

    @validate_retriever_inputs
    @type_checked
    def graph_search(self, query: str, max_hops: int = 2, limit: int = 10) -> List[PaperMetadata]:
        """
        Perform graph-based search with multi-hop reasoning.
        
        Args:
            query: Search query
            max_hops: Maximum number of hops for graph traversal (1-5)
            limit: Maximum number of results to return (1-100)
            
        Returns:
            List of relevant documents with graph context
            
        Raises:
            ValidationError: If input parameters are invalid
            SearchError: If search operation fails
            DatabaseError: If database query fails
        """
        start_time = time.time()
        
        try:
            # Validate search parameters
            search_params = validate_search_params(query, max_results=20, max_hops=max_hops, limit=limit)
            
            # Check if vector index is built
            if not self.documents or not self.document_metadata:
                raise SearchError("Vector index not built. Call build_vector_index() first.")
            
            # First, find initial relevant papers using vector similarity
            try:
                query_embedding = self.embedding_model.encode([search_params.query])
                scores, indices = self.index.search(query_embedding, k=min(20, len(self.documents)))
                
                initial_papers = [self.document_metadata[i] for i in indices[0]]
                paper_ids = [paper.paper_id for paper in initial_papers]
                
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                raise SearchError(f"Vector search failed: {e}")
            
            # Perform graph traversal to find related papers
            try:
                with self.driver.session() as session:
                    # Multi-hop graph query (without APOC)
                    cypher_query = f"""
                        MATCH (start:Paper)
                        WHERE start.paper_id IN $paper_ids
                        WITH start
                        OPTIONAL MATCH (start)-[:CITES|AUTHORED_BY|USES_METHOD|HAS_KEYWORD|COLLABORATED_WITH*1..{search_params.max_hops}]-(related)
                        RETURN DISTINCT related as node
                        LIMIT $limit
                    """
                    result = session.run(cypher_query, {"paper_ids": paper_ids, "limit": search_params.limit})
                    
                    graph_papers: List[PaperMetadata] = []
                    for record in result:
                        node = record["node"]
                        if "paper_id" in node:
                            try:
                                # Validate paper data from graph
                                paper_data = {
                                    "paper_id": node["paper_id"],
                                    "title": node["title"],
                                    "abstract": node["abstract"],
                                    "year": node["year"],
                                    "journal": node.get("journal"),
                                    "citations": node.get("citations", 0),
                                    "keywords": [],
                                    "authors": [],
                                    "methods": []
                                }
                                
                                validated_paper = Paper(**paper_data)
                                graph_papers.append(PaperMetadata(
                                    paper_id=validated_paper.paper_id,
                                    title=validated_paper.title,
                                    abstract=validated_paper.abstract,
                                    year=validated_paper.year,
                                    journal=validated_paper.journal,
                                    citations=validated_paper.citations,
                                    keywords=validated_paper.keywords,
                                    authors=validated_paper.authors,
                                    methods=validated_paper.methods
                                ))
                                
                            except Exception as e:
                                logger.warning(f"Skipping invalid paper from graph: {e}")
                                continue
                    
                    search_time = time.time() - start_time
                    self.search_times.append(search_time)
                    
                    logger.info(f"Graph search completed in {search_time:.2f}s. Found {len(graph_papers)} papers.")
                    return graph_papers
                    
            except Exception as e:
                logger.error(f"Graph traversal failed: {e}")
                raise DatabaseError(f"Graph traversal failed: {e}")
                
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            raise SearchError(f"Graph search failed: {e}")

    def get_research_gaps(self, topic: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        Identify research gaps by analyzing the knowledge graph.
        
        Args:
            topic: Research topic to analyze
            max_hops: Maximum hops for graph traversal
            
        Returns:
            Dictionary containing research gaps and opportunities
        """
        with self.driver.session() as session:
            # Find papers related to the topic
            result = session.run("""
                MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                WHERE toLower(k.text) CONTAINS toLower($topic)
                RETURN p.paper_id as paper_id, p.title as title
                LIMIT 10
            """, {"topic": topic})
            
            topic_papers = [record["paper_id"] for record in result]
            
            if not topic_papers:
                return {"error": "No papers found for the given topic"}
            
            # Analyze methodology gaps
            method_gaps = session.run("""
                MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                WHERE p.paper_id IN $papers
                OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
                WITH k.text as keyword, collect(DISTINCT m.name) as methods
                WHERE size(methods) < 3  // Papers with few methods
                RETURN keyword, methods
                LIMIT 5
            """, {"papers": topic_papers})
            
            # Analyze collaboration gaps
            collaboration_gaps = session.run("""
                MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
                WHERE p.paper_id IN $papers
                WITH a.name as author, count(p) as paper_count
                WHERE paper_count = 1  // Authors with single papers
                RETURN author, paper_count
                LIMIT 5
            """, {"papers": topic_papers})
            
            return {
                "topic": topic,
                "methodology_gaps": [{"keyword": r["keyword"], "methods": r["methods"]} 
                                   for r in method_gaps],
                "collaboration_opportunities": [{"author": r["author"], "papers": r["paper_count"]} 
                                             for r in collaboration_gaps]
            }

    def get_collaboration_network(self, author_name: str = None) -> Dict[str, Any]:
        """
        Analyze collaboration networks.
        
        Args:
            author_name: Specific author to analyze (optional)
            
        Returns:
            Collaboration network data
        """
        with self.driver.session() as session:
            if author_name:
                # Get collaboration network for specific author
                result = session.run("""
                    MATCH (a:Author {name: $author_name})-[:COLLABORATED_WITH]-(collaborator:Author)
                    RETURN collaborator.name as name, collaborator.institution as institution
                """, {"author_name": author_name})
            else:
                # Get overall collaboration statistics
                result = session.run("""
                    MATCH (a:Author)-[:COLLABORATED_WITH]-(b:Author)
                    RETURN a.name as author1, b.name as author2, 
                           a.institution as institution1, b.institution as institution2
                """)
            
            collaborations = [{"author1": r["author1"], "author2": r["author2"],
                             "institution1": r["institution1"], "institution2": r["institution2"]}
                            for r in result]
            
            return {"collaborations": collaborations}

    def get_methodology_evolution(self, method_name: str) -> Dict[str, Any]:
        """
        Track how a methodology evolves over time and across disciplines.
        
        Args:
            method_name: Name of the methodology to track
            
        Returns:
            Evolution data for the methodology
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: $method_name})
                OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                RETURN p.year as year, p.title as title, p.journal as journal,
                       collect(DISTINCT k.text) as keywords
                ORDER BY p.year
            """, {"method_name": method_name})
            
            evolution = []
            for record in result:
                evolution.append({
                    "year": record["year"],
                    "title": record["title"],
                    "journal": record["journal"],
                    "keywords": record["keywords"]
                })
            
            return {"method": method_name, "evolution": evolution}

    @validate_retriever_inputs
    @type_checked
    def hybrid_search(self, query: str, alpha: float = 0.7, limit: int = 10) -> List[PaperWithScore]:
        """
        Perform hybrid search combining vector similarity and graph structure.
        
        Args:
            query: Search query
            alpha: Weight for vector similarity (1-alpha for graph structure) (0.0-1.0)
            limit: Maximum number of results (1-100)
            
        Returns:
            List of relevant documents with hybrid scores
            
        Raises:
            ValidationError: If input parameters are invalid
            SearchError: If search operation fails
        """
        start_time = time.time()
        
        try:
            # Validate parameters
            if not 0.0 <= alpha <= 1.0:
                raise ValidationError("Alpha must be between 0.0 and 1.0")
            
            if not 1 <= limit <= 100:
                raise ValidationError("Limit must be between 1 and 100")
            
            # Check if vector index is built
            if not self.documents or not self.document_metadata:
                raise SearchError("Vector index not built. Call build_vector_index() first.")
            
            # Vector similarity search
            try:
                query_embedding = self.embedding_model.encode([query])
                vector_scores, vector_indices = self.index.search(query_embedding, k=min(20, len(self.documents)))
            except Exception as e:
                logger.error(f"Vector search failed in hybrid search: {e}")
                raise SearchError(f"Vector search failed: {e}")
            
            # Graph-based search
            try:
                graph_results = self.graph_search(query, max_hops=2, limit=limit)
            except Exception as e:
                logger.error(f"Graph search failed in hybrid search: {e}")
                raise SearchError(f"Graph search failed: {e}")
            
            # Combine scores
            hybrid_results: List[PaperWithScore] = []
            seen_papers = set()
            
            # Add vector results
            for i, idx in enumerate(vector_indices[0]):
                if idx < len(self.document_metadata):
                    doc = self.document_metadata[idx]
                    vector_score = float(vector_scores[0][i])
                    
                    hybrid_results.append(PaperWithScore(
                        paper_id=doc.paper_id,
                        title=doc.title,
                        abstract=doc.abstract,
                        year=doc.year,
                        journal=doc.journal,
                        citations=doc.citations,
                        authors=doc.authors,
                        keywords=doc.keywords,
                        methods=doc.methods,
                        vector_score=vector_score,
                        graph_score=0.0,
                        hybrid_score=alpha * vector_score
                    ))
                    seen_papers.add(doc.paper_id)
            
            # Add graph results (avoid duplicates)
            for doc in graph_results:
                if doc.paper_id not in seen_papers:
                    hybrid_results.append(PaperWithScore(
                        paper_id=doc.paper_id,
                        title=doc.title,
                        abstract=doc.abstract,
                        year=doc.year,
                        journal=doc.journal,
                        citations=doc.citations,
                        authors=doc.authors,
                        keywords=doc.keywords,
                        methods=doc.methods,
                        vector_score=0.0,
                        graph_score=1.0,
                        hybrid_score=(1 - alpha) * 1.0
                    ))
                    seen_papers.add(doc.paper_id)
            
            # Sort by hybrid score and return top results
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            result = hybrid_results[:limit]
            
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            logger.info(f"Hybrid search completed in {search_time:.2f}s. Found {len(result)} papers.")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the retriever.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.search_times:
            return {
                "total_searches": 0,
                "average_search_time": 0.0,
                "min_search_time": 0.0,
                "max_search_time": 0.0,
                "index_build_time": self.index_build_time or 0.0,
                "total_documents": len(self.documents)
            }
        
        return {
            "total_searches": len(self.search_times),
            "average_search_time": sum(self.search_times) / len(self.search_times),
            "min_search_time": min(self.search_times),
            "max_search_time": max(self.search_times),
            "index_build_time": self.index_build_time or 0.0,
            "total_documents": len(self.documents)
        }

    def validate_index_integrity(self) -> bool:
        """
        Validate the integrity of the vector index.
        
        Returns:
            True if index is valid, False otherwise
        """
        try:
            if not self.documents or not self.document_metadata:
                logger.warning("No documents in index")
                return False
            
            if len(self.documents) != len(self.document_metadata):
                logger.error("Mismatch between documents and metadata count")
                return False
            
            if self.index.ntotal != len(self.documents):
                logger.error("FAISS index size doesn't match document count")
                return False
            
            # Check for valid paper IDs
            for i, metadata in enumerate(self.document_metadata):
                if not metadata.paper_id or not metadata.title:
                    logger.warning(f"Invalid metadata at index {i}")
                    return False
            
            logger.info("Index integrity validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Index integrity validation failed: {e}")
            return False

    def rebuild_index_if_needed(self) -> bool:
        """
        Rebuild the vector index if it's corrupted or missing.
        
        Returns:
            True if rebuild was successful, False otherwise
        """
        try:
            if not self.validate_index_integrity():
                logger.info("Index validation failed, rebuilding...")
                self.build_vector_index()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            return False 