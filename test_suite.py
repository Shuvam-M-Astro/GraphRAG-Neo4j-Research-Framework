#!/usr/bin/env python3
"""
Comprehensive Test Suite for GraphRAG Neo4j Research Framework
Consolidated from multiple redundant test files.
"""

import os
import sys
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph_rag.orchestrator import GraphRAGOrchestrator
from graph_rag.graph_retriever import GraphRetriever
from validation import (
    validate_environment_variables, validate_paper_data, validate_search_params,
    validate_database_config, validate_model_config, Paper, SearchQuery,
    DatabaseConfig, ModelConfig, ValidationError
)
from types import (
    PaperId, PaperMetadata, SearchResponse, ValidationError as TypesValidationError,
    is_paper_metadata, to_paper_metadata
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self):
        """Initialize the comprehensive test suite."""
        self.orchestrator = None
        self.retriever = None
        
    def test_environment_setup(self) -> bool:
        """Test environment variable validation."""
        logger.info("Testing environment variable validation")
        
        try:
            config = validate_environment_variables()
            logger.info(f"Environment validation passed - found {len(config)} required variables")
            return True
        except ValidationError as e:
            logger.error(f"Environment validation failed: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """Test basic Neo4j connection and operations."""
        logger.info("Testing Neo4j database connection")
        
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password123")
            
            driver = GraphDatabase.driver(uri, auth=(user, password))
            
            with driver.session() as session:
                # Test basic count
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                logger.info(f"Database connection successful - {count} total nodes")
                
                # Test paper retrieval
                result = session.run("MATCH (p:Paper) RETURN p.paper_id as id, p.title as title LIMIT 3")
                papers = [record for record in result]
                logger.info(f"Retrieved {len(papers)} sample papers")
                
                # Test relationship query (no APOC)
                if papers:
                    paper_ids = [p["id"] for p in papers]
                    result = session.run("""
                        MATCH (p:Paper)
                        WHERE p.paper_id IN $paper_ids
                        OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
                        RETURN p.paper_id as paper_id, p.title as title, 
                               collect(DISTINCT k.text) as keywords
                    """, {"paper_ids": paper_ids})
                    
                    results = [record for record in result]
                    logger.info(f"Relationship query successful - {len(results)} results")
            
            driver.close()
            return True
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def test_validation_system(self) -> bool:
        """Test the validation system."""
        logger.info("Testing validation system")
        
        # Test valid paper data
        valid_paper = {
            "paper_id": "test_paper_123",
            "title": "A Comprehensive Study of Graph Neural Networks",
            "abstract": "This paper presents a comprehensive study of graph neural networks and their applications.",
            "year": 2023,
            "journal": "Nature Machine Intelligence",
            "citations": 150,
            "authors": ["Dr. Sarah Johnson", "Prof. Michael Chen"],
            "keywords": ["graph neural networks", "deep learning"],
            "methods": ["transformer", "attention mechanism"]
        }
        
        try:
            validated_paper = validate_paper_data(valid_paper)
            logger.info(f"Valid paper validation passed - {validated_paper.title}")
        except ValidationError as e:
            logger.error(f"Valid paper validation failed: {e}")
            return False
        
        # Test invalid paper data
        invalid_paper = {
            "paper_id": "",
            "title": "",
            "abstract": "Too short",
            "year": 1800,
            "citations": -5
        }
        
        try:
            validate_paper_data(invalid_paper)
            logger.error("Invalid paper validation should have failed")
            return False
        except ValidationError:
            logger.info("Invalid paper correctly rejected")
        
        # Test search parameters
        try:
            search_params = validate_search_params("test query", max_results=20, max_hops=2, limit=10)
            logger.info(f"Search parameter validation passed - {search_params.query}")
        except ValidationError as e:
            logger.error(f"Search parameter validation failed: {e}")
            return False
        
        return True
    
    def test_graph_retriever(self) -> bool:
        """Test GraphRetriever functionality."""
        logger.info("Testing GraphRetriever")
        
        try:
            self.retriever = GraphRetriever()
            logger.info("GraphRetriever initialized successfully")
            
            # Test vector search
            query_embedding = self.retriever.embedding_model.encode(["quantum computing"])
            scores, indices = self.retriever.index.search(query_embedding, k=min(5, len(self.retriever.documents)))
            logger.info(f"Vector search successful - {len(indices[0])} results")
            
            # Test hybrid search
            results = self.retriever.hybrid_search("quantum computing", limit=5)
            logger.info(f"Hybrid search successful - {len(results)} results")
            
            return True
            
        except Exception as e:
            logger.error(f"GraphRetriever test failed: {e}")
            return False
    
    def test_orchestrator(self) -> bool:
        """Test GraphRAGOrchestrator functionality."""
        logger.info("Testing GraphRAGOrchestrator")
        
        try:
            self.orchestrator = GraphRAGOrchestrator()
            logger.info("GraphRAGOrchestrator initialized successfully")
            
            # Test research topic analysis
            results = self.orchestrator.analyze_research_topic("deep learning", "comprehensive")
            
            if "error" in results:
                logger.error(f"Research topic analysis failed: {results['error']}")
                return False
            else:
                logger.info(f"Research topic analysis successful - {len(results['retrieved_papers'])} papers")
            
            # Test literature review generation
            lit_review = self.orchestrator.generate_literature_review("transformer architecture", max_papers=5)
            
            if "error" in lit_review:
                logger.error(f"Literature review generation failed: {lit_review['error']}")
                return False
            else:
                logger.info(f"Literature review generation successful - {lit_review['papers_analyzed']} papers")
            
            return True
            
        except Exception as e:
            logger.error(f"GraphRAGOrchestrator test failed: {e}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance tracking."""
        logger.info("Testing performance metrics")
        
        try:
            if self.retriever:
                stats = self.retriever.get_performance_stats()
                logger.info(f"Performance stats: {stats['total_searches']} searches, "
                           f"avg time: {stats['average_search_time']:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.retriever:
            self.retriever.close()
        if self.orchestrator:
            self.orchestrator.close()
    
    def run_all_tests(self) -> bool:
        """Run all tests in sequence."""
        logger.info("🚀 Starting Comprehensive Test Suite")
        logger.info("=" * 50)
        
        tests = [
            ("Environment Setup", self.test_environment_setup),
            ("Database Connection", self.test_database_connection),
            ("Validation System", self.test_validation_system),
            ("GraphRetriever", self.test_graph_retriever),
            ("GraphRAGOrchestrator", self.test_orchestrator),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running {test_name} test...")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} test passed")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} test failed")
            except Exception as e:
                logger.error(f"❌ {test_name} test failed with exception: {e}")
        
        logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed successfully!")
        else:
            logger.error(f"❌ {total - passed} tests failed")
        
        self.cleanup()
        return passed == total

def main():
    """Main test function."""
    load_dotenv()
    
    # Check environment variables
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set up your .env file with the required variables.")
        return False
    
    # Run comprehensive test suite
    test_suite = ComprehensiveTestSuite()
    success = test_suite.run_all_tests()
    
    if success:
        logger.info("\n✅ Comprehensive test suite completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run 'python src/database/init_database.py' to initialize the database")
        logger.info("2. Run 'python src/data_ingestion/arxiv_ingestion.py' to ingest sample data")
        logger.info("3. Run 'streamlit run src/app/main.py' to start the web application")
    else:
        logger.error("\n❌ Comprehensive test suite failed!")
        logger.error("Please check your configuration and try again.")
    
    return success

if __name__ == "__main__":
    main() 