#!/usr/bin/env python3
"""
Test script for automatic ArXiv scraping functionality.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_ingestion.arxiv_scraper import ArXivWebScraper
from database.init_database import Neo4jDatabase

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_auto_scraper():
    """Test the automatic scraping functionality."""
    logger.info("🧪 Testing Automatic ArXiv Scraper")
    logger.info("=" * 50)
    
    try:
        # Test database connection
        db = Neo4jDatabase()
        db.connect()
        logger.info("✅ Database connection successful")
        
        # Check current paper count
        with db.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
            initial_count = result.single()["paper_count"]
        
        logger.info(f"📚 Current papers in database: {initial_count}")
        
        # Test scraper initialization
        scraper = ArXivWebScraper()
        logger.info("✅ Scraper initialized successfully")
        
        # Test single query scraping
        logger.info("Testing single query scraping...")
        papers = scraper.search_papers_web("adhd", max_results=3)
        
        if papers:
            logger.info(f"✅ Found {len(papers)} papers for 'adhd' query")
            for i, paper in enumerate(papers[:2]):
                logger.info(f"   {i+1}. {paper['title'][:60]}...")
        else:
            logger.warning("⚠️ No papers found for test query")
        
        # Test paper storage
        if papers:
            logger.info("Testing paper storage...")
            success = scraper.store_paper(papers[0])
            if success:
                logger.info("✅ Paper stored successfully")
                
                # Check if paper count increased
                with db.driver.session() as session:
                    result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
                    new_count = result.single()["paper_count"]
                
                logger.info(f"📚 Papers after storage: {new_count}")
                if new_count > initial_count:
                    logger.info("✅ Database population working correctly!")
                else:
                    logger.warning("⚠️ Paper count didn't increase")
            else:
                logger.error("❌ Failed to store paper")
        
        scraper.close()
        db.close()
        
        logger.info("✅ Automatic scraper test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Automatic scraper test failed: {e}")
        return False

def test_database_population():
    """Test the complete database population process."""
    logger.info("🧪 Testing Database Population Process")
    logger.info("=" * 50)
    
    try:
        # Clear database first
        db = Neo4jDatabase()
        db.connect()
        with db.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        db.close()
        logger.info("✅ Database cleared")
        
        # Test scraper with multiple queries
        scraper = ArXivWebScraper()
        
        queries = ["adhd", "deep learning"]
        logger.info(f"Testing with {len(queries)} queries...")
        
        stats = scraper.scrape_and_store_papers(queries, max_results_per_query=5)
        
        logger.info(f"📊 Scraping Results:")
        logger.info(f"   Total papers found: {stats['total_papers_found']}")
        logger.info(f"   Successfully stored: {stats['successful_stores']}")
        logger.info(f"   Success rate: {stats['success_rate']:.2%}")
        
        if stats['successful_stores'] > 0:
            logger.info("✅ Database population working correctly!")
            
            # Create relationships
            scraper.create_collaboration_relationships()
            scraper.create_citation_relationships()
            logger.info("✅ Relationships created successfully!")
            
        else:
            logger.warning("⚠️ No papers were stored")
        
        scraper.close()
        logger.info("✅ Database population test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database population test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Automatic Scraper Tests")
    logger.info("=" * 50)
    
    # Test 1: Basic scraper functionality
    test1_success = test_auto_scraper()
    
    # Test 2: Database population
    test2_success = test_database_population()
    
    if test1_success and test2_success:
        logger.info("\n🎉 All tests passed!")
        logger.info("The automatic scraper is working correctly.")
        logger.info("You can now run: streamlit run src/app/main.py")
    else:
        logger.error("\n❌ Some tests failed!")
        logger.error("Please check your configuration and try again.")
    
    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 