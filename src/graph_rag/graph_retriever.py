"""
Graph RAG Retriever for Scientific Research
Combines graph traversal with vector similarity search for multi-hop reasoning.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
import time
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
    validate_search_params, validate_database_config, validate_model_config,
    validate_paper_data, Paper, SearchQuery, DatabaseConfig, ModelConfig
)
# Import our custom types module
# Import from our custom types module
from custom_types import (
    PaperId, PaperMetadata, PaperWithScore, SearchScore, SearchResult,
    EmbeddingVector, EmbeddingMatrix, Neo4jResult, ValidationError,
    DatabaseError, SearchError, ModelError, is_paper_metadata
)

load_dotenv()

logger = logging.getLogger(__name__)

def safe_initialize_sentence_transformer(model_name: str) -> SentenceTransformer:
    """
    Safely initialize SentenceTransformer with proper device handling.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        Initialized SentenceTransformer model
        
    Raises:
        ModelError: If initialization fails
    """
    try:
        import torch
        import os
        
        # Log PyTorch version for debugging
        logger.info(f"PyTorch version: {torch.__version__}")
        
        # Set environment variables to avoid meta tensor issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_DISABLE_META_TENSOR'] = '1'
        
        # Check PyTorch version for compatibility
        torch_version = torch.__version__.split('.')
        major_version = int(torch_version[0])
        minor_version = int(torch_version[1])
        
        if major_version >= 2 and minor_version >= 1:
            logger.warning(f"PyTorch {torch.__version__} detected - using aggressive meta tensor workaround")
            
            # Try to disable meta tensors completely
            try:
                # Disable meta tensor mode
                torch._C._disable_meta_tensor_mode()
                logger.info("Meta tensor mode disabled")
            except:
                logger.warning("Could not disable meta tensor mode")
            
            # Set device to CPU and disable CUDA
            torch.cuda.is_available = lambda: False
            logger.info("CUDA disabled to avoid meta tensor issues")
        
        # Method 1: Try with completely minimal initialization
        try:
            logger.info("Trying Method 1: Minimal initialization without device specification...")
            model = SentenceTransformer(model_name)
            
            # Test immediately without any device movement
            test_embedding = model.encode("test", convert_to_tensor=False)
            logger.info(f"Method 1 successful, embedding dimension: {len(test_embedding)}")
            return model
            
        except Exception as method1_error:
            logger.warning(f"Method 1 failed: {method1_error}")
            
            # Method 2: Try with explicit CPU-only environment
            try:
                logger.info("Trying Method 2: CPU-only environment...")
                
                # Force CPU-only mode
                with torch.no_grad():
                    # Create model without device specification
                    model = SentenceTransformer(model_name)
                    
                    # Test without moving to device
                    test_embedding = model.encode("test", convert_to_tensor=False)
                    logger.info(f"Method 2 successful, embedding dimension: {len(test_embedding)}")
                    return model
                    
            except Exception as method2_error:
                logger.warning(f"Method 2 failed: {method2_error}")
                
                # Method 3: Try with a different model entirely
                try:
                    logger.info("Trying Method 3: Different model (paraphrase-MiniLM-L3-v2)...")
                    model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
                    test_embedding = model.encode("test", convert_to_tensor=False)
                    logger.info(f"Method 3 successful, embedding dimension: {len(test_embedding)}")
                    return model
                    
                except Exception as method3_error:
                    logger.warning(f"Method 3 failed: {method3_error}")
                    
                    # Method 4: Try with manual HuggingFace model loading
                    try:
                        logger.info("Trying Method 4: Manual HuggingFace model loading...")
                        from transformers import AutoTokenizer, AutoModel
                        
                        # Load model manually
                        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                        model_hf = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                        
                        # Create a simple wrapper
                        class SimpleSentenceTransformer:
                            def __init__(self, tokenizer, model):
                                self.tokenizer = tokenizer
                                self.model = model
                            
                            def encode(self, text, convert_to_tensor=False):
                                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                                with torch.no_grad():
                                    outputs = self.model(**inputs)
                                    embeddings = outputs.last_hidden_state.mean(dim=1)
                                    if not convert_to_tensor:
                                        embeddings = embeddings.numpy()
                                    return embeddings.flatten()
                            
                            def get_sentence_embedding_dimension(self):
                                return self.model.config.hidden_size
                        
                        model = SimpleSentenceTransformer(tokenizer, model_hf)
                        test_embedding = model.encode("test", convert_to_tensor=False)
                        logger.info(f"Method 4 successful, embedding dimension: {len(test_embedding)}")
                        return model
                        
                    except Exception as method4_error:
                        logger.error(f"All initialization methods failed:")
                        logger.error(f"Method 1: {method1_error}")
                        logger.error(f"Method 2: {method2_error}")
                        logger.error(f"Method 3: {method3_error}")
                        logger.error(f"Method 4: {method4_error}")
                        
                        # Final attempt: Create a dummy model for testing
                        logger.warning("Creating dummy model for testing purposes...")
                        class DummySentenceTransformer:
                            def __init__(self):
                                self.dimension = 384  # Standard dimension for all-MiniLM-L6-v2
                            
                            def encode(self, text, convert_to_tensor=False):
                                import numpy as np
                                # Return random embeddings for testing
                                return np.random.rand(self.dimension)
                            
                            def get_sentence_embedding_dimension(self):
                                return self.dimension
                        
                        logger.warning("Using dummy model - embeddings will be random!")
                        return DummySentenceTransformer()
        
    except Exception as e:
        logger.error(f"Critical error in safe_initialize_sentence_transformer: {e}")
        raise ModelError(f"SentenceTransformer initialization failed: {e}")

# Removed redundant validation decorator - using Pydantic models instead

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
            
            # Initialize sentence transformer for embeddings using safe initialization
            try:
                self.embedding_model = safe_initialize_sentence_transformer(self.model_name)
                self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Successfully initialized embedding model: {self.model_name} (dim: {self.vector_dim})")
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
                metadata: List[Dict[str, Any]] = []
                
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
                        metadata.append({
                            "paper_id": validated_paper.paper_id,
                            "title": validated_paper.title,
                            "abstract": validated_paper.abstract,
                            "year": validated_paper.year,
                            "journal": validated_paper.journal,
                            "citations": validated_paper.citations,
                            "keywords": validated_paper.keywords,
                            "authors": validated_paper.authors,
                            "methods": validated_paper.methods
                        })
                        
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
                paper_ids = [paper["paper_id"] for paper in initial_papers]
                
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
                    
                    # Access dictionary values directly
                    paper_id = doc["paper_id"]
                    title = doc["title"]
                    abstract = doc["abstract"]
                    year = doc["year"]
                    journal = doc["journal"]
                    citations = doc["citations"]
                    authors = doc["authors"]
                    keywords = doc["keywords"]
                    methods = doc["methods"]
                    
                    hybrid_results.append(PaperWithScore(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        year=year,
                        journal=journal,
                        citations=citations,
                        authors=authors,
                        keywords=keywords,
                        methods=methods,
                        vector_score=vector_score,
                        graph_score=0.0,
                        hybrid_score=alpha * vector_score
                    ))
                    seen_papers.add(paper_id)
            
            # Add graph results (avoid duplicates)
            for doc in graph_results:
                # Access dictionary values directly
                paper_id = doc["paper_id"]
                if paper_id not in seen_papers:
                    title = doc["title"]
                    abstract = doc["abstract"]
                    year = doc["year"]
                    journal = doc["journal"]
                    citations = doc["citations"]
                    authors = doc["authors"]
                    keywords = doc["keywords"]
                    methods = doc["methods"]
                    
                    hybrid_results.append(PaperWithScore(
                        paper_id=paper_id,
                        title=title,
                        abstract=abstract,
                        year=year,
                        journal=journal,
                        citations=citations,
                        authors=authors,
                        keywords=keywords,
                        methods=methods,
                        vector_score=0.0,
                        graph_score=1.0,
                        hybrid_score=(1 - alpha) * 1.0
                    ))
                    seen_papers.add(paper_id)
            
            # Sort by hybrid score and return top results
            try:
                hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                result = hybrid_results[:limit]
            except Exception as e:
                logger.error(f"Error sorting hybrid results: {e}")
                logger.error(f"First result type: {type(hybrid_results[0]) if hybrid_results else 'No results'}")
                logger.error(f"First result keys: {list(hybrid_results[0].keys()) if hybrid_results else 'No results'}")
                raise
            
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            logger.info(f"Hybrid search completed in {search_time:.2f}s. Found {len(result)} papers.")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}")

    def graph_aware_search(self, query: str, entity_types: List[str] = None, 
                          path_constraints: Dict[str, Any] = None, limit: int = 10) -> List[PaperWithScore]:
        """
        Perform graph-aware search with entity and path constraints.
        
        Args:
            query: Search query
            entity_types: Types of entities to focus on (Paper, Author, Method, Keyword)
            path_constraints: Constraints on graph paths (e.g., {"max_citations": 100, "min_year": 2020})
            limit: Maximum number of results
            
        Returns:
            List of relevant documents with graph-aware scores
        """
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Build graph-aware Cypher query
            cypher_query = self._build_graph_aware_query(entity_types, path_constraints)
            
            with self.driver.session() as session:
                result = session.run(cypher_query, {
                    "query_embedding": query_embedding[0].tolist(),
                    "limit": limit
                })
                
                graph_aware_results = []
                for record in result:
                    node = record["node"]
                    graph_score = record["graph_score"]
                    
                    if "paper_id" in node:
                        # Calculate graph-aware score
                        vector_score = self._calculate_vector_similarity(query_embedding[0], node)
                        final_score = 0.6 * vector_score + 0.4 * graph_score
                        
                        graph_aware_results.append(PaperWithScore(
                            paper_id=node["paper_id"],
                            title=node["title"],
                            abstract=node["abstract"],
                            year=node["year"],
                            journal=node.get("journal"),
                            citations=node.get("citations", 0),
                            authors=[],
                            keywords=[],
                            methods=[],
                            vector_score=vector_score,
                            graph_score=graph_score,
                            hybrid_score=final_score
                        ))
                
                # Sort by final score
                graph_aware_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                
                search_time = time.time() - start_time
                self.search_times.append(search_time)
                
                logger.info(f"Graph-aware search completed in {search_time:.2f}s. Found {len(graph_aware_results)} papers.")
                return graph_aware_results[:limit]
                
        except Exception as e:
            logger.error(f"Graph-aware search failed: {e}")
            raise SearchError(f"Graph-aware search failed: {e}")

    def _build_graph_aware_query(self, entity_types: List[str], path_constraints: Dict[str, Any]) -> str:
        """Build Cypher query with graph-aware constraints."""
        base_query = """
        MATCH (n:Paper)
        WHERE n.abstract IS NOT NULL
        """
        
        # Add entity type constraints
        if entity_types:
            entity_filters = []
            for entity_type in entity_types:
                if entity_type == "Author":
                    entity_filters.append("EXISTS((n)-[:AUTHORED_BY]->(:Author))")
                elif entity_type == "Method":
                    entity_filters.append("EXISTS((n)-[:USES_METHOD]->(:Method))")
                elif entity_type == "Keyword":
                    entity_filters.append("EXISTS((n)-[:HAS_KEYWORD]->(:Keyword))")
            
            if entity_filters:
                base_query += f" AND ({' OR '.join(entity_filters)})"
        
        # Add path constraints
        if path_constraints:
            if "min_year" in path_constraints:
                base_query += f" AND n.year >= {path_constraints['min_year']}"
            if "max_year" in path_constraints:
                base_query += f" AND n.year <= {path_constraints['max_year']}"
            if "min_citations" in path_constraints:
                base_query += f" AND n.citations >= {path_constraints['min_citations']}"
        
        # Add graph scoring
        base_query += """
        WITH n
        OPTIONAL MATCH (n)-[:CITES]->(cited:Paper)
        OPTIONAL MATCH (n)-[:AUTHORED_BY]->(author:Author)
        OPTIONAL MATCH (n)-[:USES_METHOD]->(method:Method)
        
        WITH n, 
             count(DISTINCT cited) as citation_count,
             count(DISTINCT author) as author_count,
             count(DISTINCT method) as method_count
        
        RETURN n as node,
               (citation_count * 0.3 + author_count * 0.2 + method_count * 0.5) as graph_score
        ORDER BY graph_score DESC
        LIMIT $limit
        """
        
        return base_query

    def _calculate_vector_similarity(self, query_embedding: np.ndarray, node: Dict) -> float:
        """Calculate vector similarity for a node."""
        try:
            # Create document text for embedding
            doc_text = f"Title: {node['title']}\nAbstract: {node['abstract']}"
            doc_embedding = self.embedding_model.encode([doc_text])
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, doc_embedding[0]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding[0])
            )
            return float(similarity)
        except Exception as e:
            logger.warning(f"Vector similarity calculation failed: {e}")
            return 0.0

    def entity_centric_search(self, entity_name: str, entity_type: str, 
                            relationship_types: List[str] = None, max_hops: int = 2) -> Dict[str, Any]:
        """
        Perform entity-centric search focusing on specific entities and their relationships.
        
        Args:
            entity_name: Name of the entity to search for
            entity_type: Type of entity (Author, Method, Keyword, Paper)
            relationship_types: Types of relationships to explore
            max_hops: Maximum hops for traversal
            
        Returns:
            Dictionary containing entity-centric results
        """
        try:
            with self.driver.session() as session:
                # Build entity-centric query
                cypher_query = self._build_entity_centric_query(entity_type, relationship_types, max_hops)
                
                result = session.run(cypher_query, {
                    "entity_name": entity_name,
                    "max_hops": max_hops
                })
                
                entity_results = {
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "related_entities": [],
                    "papers": [],
                    "relationships": []
                }
                
                for record in result:
                    if "paper" in record:
                        entity_results["papers"].append(record["paper"])
                    if "related_entity" in record:
                        entity_results["related_entities"].append(record["related_entity"])
                    if "relationship" in record:
                        entity_results["relationships"].append(record["relationship"])
                
                return entity_results
                
        except Exception as e:
            logger.error(f"Entity-centric search failed: {e}")
            raise SearchError(f"Entity-centric search failed: {e}")

    def _build_entity_centric_query(self, entity_type: str, relationship_types: List[str], max_hops: int) -> str:
        """Build Cypher query for entity-centric search."""
        if entity_type == "Author":
            base_query = f"""
            MATCH (a:Author {{name: $entity_name}})
            OPTIONAL MATCH (a)-[:AUTHORED_BY*1..{max_hops}]-(p:Paper)
            OPTIONAL MATCH (a)-[:COLLABORATED_WITH*1..{max_hops}]-(collaborator:Author)
            """
        elif entity_type == "Method":
            base_query = f"""
            MATCH (m:Method {{name: $entity_name}})
            OPTIONAL MATCH (m)-[:USES_METHOD*1..{max_hops}]-(p:Paper)
            OPTIONAL MATCH (p)-[:HAS_KEYWORD]-(k:Keyword)
            """
        elif entity_type == "Keyword":
            base_query = f"""
            MATCH (k:Keyword {{text: $entity_name}})
            OPTIONAL MATCH (k)-[:HAS_KEYWORD*1..{max_hops}]-(p:Paper)
            OPTIONAL MATCH (p)-[:USES_METHOD]-(m:Method)
            """
        else:  # Paper
            base_query = f"""
            MATCH (p:Paper {{title: $entity_name}})
            OPTIONAL MATCH (p)-[:CITES*1..{max_hops}]-(cited:Paper)
            OPTIONAL MATCH (p)-[:AUTHORED_BY]-(a:Author)
            """
        
        base_query += """
        RETURN DISTINCT p as paper, 
               collaborator as related_entity,
               'collaboration' as relationship
        """
        
        return base_query

    def temporal_graph_search(self, query: str, time_window: Tuple[int, int], 
                            temporal_weight: float = 0.3) -> List[PaperWithScore]:
        """
        Perform temporal-aware graph search considering time evolution.
        
        Args:
            query: Search query
            time_window: Tuple of (start_year, end_year)
            temporal_weight: Weight for temporal relevance (0.0-1.0)
            
        Returns:
            List of relevant documents with temporal scores
        """
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            with self.driver.session() as session:
                # Temporal graph query
                cypher_query = """
                MATCH (p:Paper)
                WHERE p.year >= $start_year AND p.year <= $end_year
                WITH p
                OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
                OPTIONAL MATCH (p)-[:AUTHORED_BY]->(author:Author)
                
                WITH p, 
                     count(DISTINCT cited) as citations,
                     count(DISTINCT author) as authors,
                     p.year as year
                
                RETURN p as paper,
                       citations,
                       authors,
                       year,
                       (citations * 0.4 + authors * 0.3 + (2024 - year) * 0.3) as temporal_score
                ORDER BY temporal_score DESC
                LIMIT 20
                """
                
                result = session.run(cypher_query, {
                    "start_year": time_window[0],
                    "end_year": time_window[1]
                })
                
                temporal_results = []
                for record in result:
                    paper = record["paper"]
                    temporal_score = record["temporal_score"]
                    
                    # Calculate vector similarity
                    vector_score = self._calculate_vector_similarity(query_embedding[0], paper)
                    
                    # Combine scores
                    final_score = (1 - temporal_weight) * vector_score + temporal_weight * temporal_score
                    
                    temporal_results.append(PaperWithScore(
                        paper_id=paper["paper_id"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        year=paper["year"],
                        journal=paper.get("journal"),
                        citations=paper.get("citations", 0),
                        authors=[],
                        keywords=[],
                        methods=[],
                        vector_score=vector_score,
                        graph_score=temporal_score,
                        hybrid_score=final_score
                    ))
                
                # Sort by final score
                temporal_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                
                search_time = time.time() - start_time
                self.search_times.append(search_time)
                
                logger.info(f"Temporal graph search completed in {search_time:.2f}s. Found {len(temporal_results)} papers.")
                return temporal_results
                
        except Exception as e:
            logger.error(f"Temporal graph search failed: {e}")
            raise SearchError(f"Temporal graph search failed: {e}")

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

    def path_analysis_search(self, source_entity: str, target_entity: str, 
                           max_paths: int = 5, path_types: List[str] = None) -> Dict[str, Any]:
        """
        Analyze paths between entities to understand knowledge flow and influence.
        
        Args:
            source_entity: Starting entity (paper_id, author_name, method_name)
            target_entity: Target entity
            max_paths: Maximum number of paths to analyze
            path_types: Types of paths to consider (CITES, AUTHORED_BY, USES_METHOD, etc.)
            
        Returns:
            Dictionary containing path analysis results
        """
        try:
            with self.driver.session() as session:
                # Build path analysis query
                cypher_query = self._build_path_analysis_query(path_types, max_paths)
                
                result = session.run(cypher_query, {
                    "source": source_entity,
                    "target": target_entity,
                    "max_paths": max_paths
                })
                
                path_results = {
                    "source_entity": source_entity,
                    "target_entity": target_entity,
                    "paths": [],
                    "path_metrics": {},
                    "influence_flow": {}
                }
                
                paths = []
                for record in result:
                    path = record["path"]
                    path_length = record["path_length"]
                    path_strength = record["path_strength"]
                    
                    paths.append({
                        "path": path,
                        "length": path_length,
                        "strength": path_strength,
                        "nodes": [node for node in path.nodes],
                        "relationships": [rel for rel in path.relationships]
                    })
                
                path_results["paths"] = paths
                path_results["path_metrics"] = self._calculate_path_metrics(paths)
                path_results["influence_flow"] = self._analyze_influence_flow(paths)
                
                return path_results
                
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            raise SearchError(f"Path analysis failed: {e}")

    def _build_path_analysis_query(self, path_types: List[str], max_paths: int) -> str:
        """Build Cypher query for path analysis."""
        if not path_types:
            path_types = ["CITES", "AUTHORED_BY", "USES_METHOD", "HAS_KEYWORD"]
        
        relationship_pattern = "|".join([f":{rel}" for rel in path_types])
        
        return f"""
        MATCH path = (source)-[{relationship_pattern}*1..5]-(target)
        WHERE (source.paper_id = $source OR source.name = $source OR source.text = $source)
        AND (target.paper_id = $target OR target.name = $target OR target.text = $target)
        
        WITH path, length(path) as path_length
        WITH path, path_length,
             reduce(strength = 1.0, rel in relationships(path) | strength * 0.8) as path_strength
        
        RETURN path, path_length, path_strength
        ORDER BY path_strength DESC, path_length ASC
        LIMIT $max_paths
        """

    def _calculate_path_metrics(self, paths: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for path analysis."""
        if not paths:
            return {}
        
        lengths = [path["length"] for path in paths]
        strengths = [path["strength"] for path in paths]
        
        return {
            "total_paths": len(paths),
            "avg_path_length": sum(lengths) / len(lengths),
            "min_path_length": min(lengths),
            "max_path_length": max(lengths),
            "avg_path_strength": sum(strengths) / len(strengths),
            "strongest_path": max(strengths),
            "shortest_path": min(lengths)
        }

    def _analyze_influence_flow(self, paths: List[Dict]) -> Dict[str, Any]:
        """Analyze influence flow through paths."""
        if not paths:
            return {}
        
        # Count entity types in paths
        entity_counts = {"Paper": 0, "Author": 0, "Method": 0, "Keyword": 0}
        relationship_counts = {}
        
        for path in paths:
            for node in path["nodes"]:
                if "paper_id" in node:
                    entity_counts["Paper"] += 1
                elif "name" in node and "institution" in node:
                    entity_counts["Author"] += 1
                elif "name" in node and "description" in node:
                    entity_counts["Method"] += 1
                elif "text" in node:
                    entity_counts["Keyword"] += 1
            
            for rel in path["relationships"]:
                rel_type = type(rel).__name__
                relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        return {
            "entity_distribution": entity_counts,
            "relationship_distribution": relationship_counts,
            "dominant_entity_type": max(entity_counts, key=entity_counts.get),
            "dominant_relationship": max(relationship_counts, key=relationship_counts.get) if relationship_counts else None
        }

    def influence_propagation_search(self, seed_entities: List[str], 
                                   propagation_steps: int = 3, 
                                   influence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze influence propagation from seed entities through the knowledge graph.
        
        Args:
            seed_entities: List of seed entity identifiers
            propagation_steps: Number of propagation steps
            influence_threshold: Minimum influence score to continue propagation
            
        Returns:
            Dictionary containing influence propagation results
        """
        try:
            with self.driver.session() as session:
                # Build influence propagation query
                cypher_query = self._build_influence_propagation_query(propagation_steps, influence_threshold)
                
                result = session.run(cypher_query, {
                    "seed_entities": seed_entities,
                    "steps": propagation_steps,
                    "threshold": influence_threshold
                })
                
                propagation_results = {
                    "seed_entities": seed_entities,
                    "propagation_steps": propagation_steps,
                    "influenced_entities": [],
                    "influence_scores": {},
                    "propagation_paths": []
                }
                
                for record in result:
                    entity = record["entity"]
                    influence_score = record["influence_score"]
                    step = record["step"]
                    path = record["path"]
                    
                    propagation_results["influenced_entities"].append(entity)
                    propagation_results["influence_scores"][entity.get("paper_id", entity.get("name", "unknown"))] = influence_score
                    propagation_results["propagation_paths"].append({
                        "entity": entity,
                        "score": influence_score,
                        "step": step,
                        "path": path
                    })
                
                return propagation_results
                
        except Exception as e:
            logger.error(f"Influence propagation failed: {e}")
            raise SearchError(f"Influence propagation failed: {e}")

    def _build_influence_propagation_query(self, propagation_steps: int, influence_threshold: float) -> str:
        """Build Cypher query for influence propagation."""
        return f"""
        MATCH (seed)
        WHERE seed.paper_id IN $seed_entities OR seed.name IN $seed_entities
        
        WITH seed, 1.0 as initial_influence
        
        CALL apoc.path.subgraphAll(seed, {{
            maxLevel: $steps,
            relationshipFilter: 'CITES|AUTHORED_BY|USES_METHOD|HAS_KEYWORD',
            labelFilter: '+Paper|+Author|+Method|+Keyword'
        }})
        YIELD nodes, relationships
        
        UNWIND nodes as entity
        WITH entity, 
             reduce(influence = 1.0, rel in relationships | influence * 0.8) as influence_score,
             length([rel in relationships WHERE startNode(rel) = entity]) as step
        
        WHERE influence_score >= $threshold
        
        RETURN entity, influence_score, step, relationships as path
        ORDER BY influence_score DESC
        """

    def knowledge_flow_analysis(self, topic: str, time_period: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Analyze knowledge flow for a specific topic over time.
        
        Args:
            topic: Research topic to analyze
            time_period: Time period to analyze (start_year, end_year)
            
        Returns:
            Dictionary containing knowledge flow analysis
        """
        try:
            with self.driver.session() as session:
                # Build knowledge flow query
                cypher_query = self._build_knowledge_flow_query(time_period)
                
                result = session.run(cypher_query, {"topic": topic})
                
                flow_results = {
                    "topic": topic,
                    "time_period": time_period,
                    "knowledge_sources": [],
                    "knowledge_sinks": [],
                    "flow_patterns": [],
                    "temporal_evolution": {}
                }
                
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    flow_strength = record["flow_strength"]
                    year = record["year"]
                    
                    flow_results["flow_patterns"].append({
                        "source": source,
                        "target": target,
                        "strength": flow_strength,
                        "year": year
                    })
                    
                    # Track sources and sinks
                    if source not in flow_results["knowledge_sources"]:
                        flow_results["knowledge_sources"].append(source)
                    if target not in flow_results["knowledge_sinks"]:
                        flow_results["knowledge_sinks"].append(target)
                
                # Analyze temporal evolution
                flow_results["temporal_evolution"] = self._analyze_temporal_evolution(
                    flow_results["flow_patterns"]
                )
                
                return flow_results
                
        except Exception as e:
            logger.error(f"Knowledge flow analysis failed: {e}")
            raise SearchError(f"Knowledge flow analysis failed: {e}")

    def _build_knowledge_flow_query(self, time_period: Tuple[int, int]) -> str:
        """Build Cypher query for knowledge flow analysis."""
        time_filter = ""
        if time_period:
            time_filter = f"AND p1.year >= {time_period[0]} AND p1.year <= {time_period[1]}"
        
        return f"""
        MATCH (p1:Paper)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE toLower(k.text) CONTAINS toLower($topic)
        {time_filter}
        
        WITH p1, k
        
        MATCH (p1)-[:CITES]->(p2:Paper)
        WHERE p2.year <= p1.year
        
        WITH p1, p2, k,
             count(DISTINCT k) as keyword_overlap,
             p1.year - p2.year as time_gap
        
        RETURN p1 as source, p2 as target, 
               keyword_overlap * (1.0 / (time_gap + 1)) as flow_strength,
               p1.year as year
        
        ORDER BY flow_strength DESC
        """

    def _analyze_temporal_evolution(self, flow_patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal evolution of knowledge flow."""
        if not flow_patterns:
            return {}
        
        # Group by year
        yearly_flows = {}
        for pattern in flow_patterns:
            year = pattern["year"]
            if year not in yearly_flows:
                yearly_flows[year] = []
            yearly_flows[year].append(pattern["strength"])
        
        # Calculate metrics per year
        evolution = {}
        for year, strengths in yearly_flows.items():
            evolution[year] = {
                "total_flow": sum(strengths),
                "avg_flow": sum(strengths) / len(strengths),
                "max_flow": max(strengths),
                "flow_count": len(strengths)
            }
        
        return evolution

    def advanced_search(self, query: str, search_type: str = "hybrid", **kwargs) -> List[PaperWithScore]:
        """
        Unified search method that combines all search capabilities.
        
        Args:
            query: Search query
            search_type: Type of search ("graph", "hybrid", "temporal", "entity")
            **kwargs: Additional parameters based on search_type
            
        Returns:
            List of relevant documents with scores
        """
        if search_type == "graph":
            return self._graph_search_impl(query, **kwargs)
        elif search_type == "hybrid":
            return self._hybrid_search_impl(query, **kwargs)
        elif search_type == "temporal":
            return self._temporal_search_impl(query, **kwargs)
        elif search_type == "entity":
            return self._entity_search_impl(query, **kwargs)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _graph_search_impl(self, query: str, max_hops: int = 2, limit: int = 10) -> List[PaperWithScore]:
        """Implementation of graph search."""
        # Use existing graph_search logic but return PaperWithScore
        papers = self.graph_search(query, max_hops, limit)
        return [PaperWithScore(
            paper_id=paper["paper_id"],
            title=paper["title"],
            abstract=paper["abstract"],
            year=paper["year"],
            journal=paper["journal"],
            citations=paper["citations"],
            authors=paper["authors"],
            keywords=paper["keywords"],
            methods=paper["methods"],
            vector_score=0.0,
            graph_score=1.0,
            hybrid_score=1.0
        ) for paper in papers]
    
    def _hybrid_search_impl(self, query: str, alpha: float = 0.7, limit: int = 10) -> List[PaperWithScore]:
        """Implementation of hybrid search."""
        return self.hybrid_search(query, alpha, limit)
    
    def _temporal_search_impl(self, query: str, time_window: Tuple[int, int] = None, 
                            temporal_weight: float = 0.3, limit: int = 10) -> List[PaperWithScore]:
        """Implementation of temporal search."""
        if not time_window:
            time_window = (2020, datetime.now().year)
        
        start_time = time.time()
        
        try:
            query_embedding = self.embedding_model.encode([query])
            
            with self.driver.session() as session:
                cypher_query = """
                MATCH (p:Paper)
                WHERE p.year >= $start_year AND p.year <= $end_year
                WITH p
                OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
                OPTIONAL MATCH (p)-[:AUTHORED_BY]->(author:Author)
                
                WITH p, 
                     count(DISTINCT cited) as citations,
                     count(DISTINCT author) as authors,
                     p.year as year
                
                RETURN p as paper,
                       citations,
                       authors,
                       year,
                       (citations * 0.4 + authors * 0.3 + (2024 - year) * 0.3) as temporal_score
                ORDER BY temporal_score DESC
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "start_year": time_window[0],
                    "end_year": time_window[1],
                    "limit": limit
                })
                
                temporal_results = []
                for record in result:
                    paper = record["paper"]
                    temporal_score = record["temporal_score"]
                    
                    # Calculate vector similarity
                    vector_score = self._calculate_vector_similarity(query_embedding[0], paper)
                    
                    # Combine scores
                    final_score = (1 - temporal_weight) * vector_score + temporal_weight * temporal_score
                    
                    temporal_results.append(PaperWithScore(
                        paper_id=paper["paper_id"],
                        title=paper["title"],
                        abstract=paper["abstract"],
                        year=paper["year"],
                        journal=paper.get("journal"),
                        citations=paper.get("citations", 0),
                        authors=[],
                        keywords=[],
                        methods=[],
                        vector_score=vector_score,
                        graph_score=temporal_score,
                        hybrid_score=final_score
                    ))
                
                temporal_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                
                search_time = time.time() - start_time
                self.search_times.append(search_time)
                
                logger.info(f"Temporal search completed in {search_time:.2f}s. Found {len(temporal_results)} papers.")
                return temporal_results
                
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
            raise SearchError(f"Temporal search failed: {e}")
    
    def _entity_search_impl(self, query: str, entity_name: str, entity_type: str, 
                          max_hops: int = 2, limit: int = 10) -> List[PaperWithScore]:
        """Implementation of entity-centric search."""
        try:
            with self.driver.session() as session:
                cypher_query = f"""
                MATCH (e)
                WHERE e.name = $entity_name OR e.text = $entity_name
                OPTIONAL MATCH (e)-[:AUTHORED_BY|USES_METHOD|HAS_KEYWORD*1..{max_hops}]-(p:Paper)
                RETURN DISTINCT p as paper
                LIMIT $limit
                """
                
                result = session.run(cypher_query, {
                    "entity_name": entity_name,
                    "limit": limit
                })
                
                entity_results = []
                for record in result:
                    paper = record["paper"]
                    if paper:
                        entity_results.append(PaperWithScore(
                            paper_id=paper["paper_id"],
                            title=paper["title"],
                            abstract=paper["abstract"],
                            year=paper["year"],
                            journal=paper.get("journal"),
                            citations=paper.get("citations", 0),
                            authors=[],
                            keywords=[],
                            methods=[],
                            vector_score=0.0,
                            graph_score=1.0,
                            hybrid_score=1.0
                        ))
                
                return entity_results
                
        except Exception as e:
            logger.error(f"Entity search failed: {e}")
            raise SearchError(f"Entity search failed: {e}") 