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

load_dotenv()

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Graph RAG retriever.
        
        Args:
            model_name: Sentence transformer model for embeddings
        """
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index for vector search
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.documents = []
        self.document_metadata = []
        
        self.connect()
        self.build_vector_index()

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

    def build_vector_index(self):
        """Build FAISS index from documents in Neo4j."""
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
            
            documents = []
            metadata = []
            
            for record in result:
                paper = record["p"]
                keywords = record["keywords"]
                authors = record["authors"]
                methods = record["methods"]
                
                # Create document text
                doc_text = f"Title: {paper['title']}\n"
                doc_text += f"Abstract: {paper['abstract']}\n"
                if keywords:
                    doc_text += f"Keywords: {', '.join(keywords)}\n"
                if authors:
                    doc_text += f"Authors: {', '.join(authors)}\n"
                if methods:
                    doc_text += f"Methods: {', '.join(methods)}\n"
                
                documents.append(doc_text)
                metadata.append({
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "year": paper["year"],
                    "journal": paper["journal"],
                    "citations": paper["citations"],
                    "keywords": keywords,
                    "authors": authors,
                    "methods": methods
                })
            
            # Create embeddings and add to FAISS index
            if documents:
                embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
                self.index.add(embeddings.astype('float32'))
                self.documents = documents
                self.document_metadata = metadata
                logger.info(f"Built vector index with {len(documents)} documents")

    def graph_search(self, query: str, max_hops: int = 2, limit: int = 10) -> List[Dict]:
        """
        Perform graph-based search with multi-hop reasoning.
        
        Args:
            query: Search query
            max_hops: Maximum number of hops for graph traversal
            limit: Maximum number of results to return
            
        Returns:
            List of relevant documents with graph context
        """
        # First, find initial relevant papers using vector similarity
        query_embedding = self.embedding_model.encode([query])
        scores, indices = self.index.search(query_embedding, k=min(20, len(self.documents)))
        
        initial_papers = [self.document_metadata[i] for i in indices[0]]
        paper_ids = [paper["paper_id"] for paper in initial_papers]
        
        # Perform graph traversal to find related papers
        with self.driver.session() as session:
            # Multi-hop graph query (without APOC)
            query = f"""
                MATCH (start:Paper)
                WHERE start.paper_id IN $paper_ids
                WITH start
                OPTIONAL MATCH (start)-[:CITES|AUTHORED_BY|USES_METHOD|HAS_KEYWORD|COLLABORATED_WITH*1..{max_hops}]-(related)
                RETURN DISTINCT related as node
                LIMIT $limit
            """
            result = session.run(query, {"paper_ids": paper_ids, "limit": limit})
            
            graph_papers = []
            for record in result:
                node = record["node"]
                if "paper_id" in node:
                    graph_papers.append({
                        "paper_id": node["paper_id"],
                        "title": node["title"],
                        "abstract": node["abstract"],
                        "year": node["year"],
                        "journal": node["journal"],
                        "citations": node["citations"]
                    })
            
            return graph_papers

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

    def hybrid_search(self, query: str, alpha: float = 0.7, limit: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining vector similarity and graph structure.
        
        Args:
            query: Search query
            alpha: Weight for vector similarity (1-alpha for graph structure)
            limit: Maximum number of results
            
        Returns:
            List of relevant documents with hybrid scores
        """
        # Vector similarity search
        query_embedding = self.embedding_model.encode([query])
        vector_scores, vector_indices = self.index.search(query_embedding, k=min(20, len(self.documents)))
        
        # Graph-based search
        graph_results = self.graph_search(query, max_hops=2, limit=limit)
        
        # Combine scores (simplified approach)
        hybrid_results = []
        
        # Add vector results
        for i, idx in enumerate(vector_indices[0]):
            if idx < len(self.document_metadata):
                doc = self.document_metadata[idx].copy()
                doc["vector_score"] = float(vector_scores[0][i])
                doc["graph_score"] = 0.0
                doc["hybrid_score"] = alpha * doc["vector_score"]
                hybrid_results.append(doc)
        
        # Add graph results
        for doc in graph_results:
            doc["vector_score"] = 0.0
            doc["graph_score"] = 1.0
            doc["hybrid_score"] = (1 - alpha) * doc["graph_score"]
            hybrid_results.append(doc)
        
        # Sort by hybrid score and return top results
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:limit] 