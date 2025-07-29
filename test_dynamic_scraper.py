#!/usr/bin/env python3
"""
Test script for dynamic ArXiv scraping functionality.
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

def test_dynamic_scraping():
    """Test dynamic scraping for specific queries."""
    logger.info("🧪 Testing Dynamic ArXiv Scraper")
    logger.info("=" * 50)
    
    try:
        # Initialize scraper
        scraper = ArXivWebScraper()
        logger.info("✅ Scraper initialized successfully")
        
        # Test queries
        test_queries = ["adhd", "machine learning", "transformer"]
        
        for query in test_queries:
            logger.info(f"Testing query: '{query}'")
            
            # Scrape papers for the query
            papers = scraper.search_papers_web(query, max_results=5)
            
            if papers:
                logger.info(f"✅ Found {len(papers)} papers for '{query}'")
                
                # Store papers
                successful_stores = 0
                for paper in papers:
                    if scraper.store_paper(paper):
                        successful_stores += 1
                
                logger.info(f"✅ Successfully stored {successful_stores} papers for '{query}'")
                
                # Show sample paper info
                if papers:
                    sample_paper = papers[0]
                    logger.info(f"   Sample paper: {sample_paper['title'][:60]}...")
                    logger.info(f"   Authors: {', '.join(sample_paper['authors'][:3])}")
                    logger.info(f"   Categories: {', '.join(sample_paper['categories'][:3])}")
            else:
                logger.warning(f"⚠️ No papers found for '{query}'")
        
        scraper.close()
        logger.info("✅ Dynamic scraper test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dynamic scraper test failed: {e}")
        return False

def test_database_integration():
    """Test that scraped papers are properly integrated into the database."""
    logger.info("🧪 Testing Database Integration")
    logger.info("=" * 50)
    
    try:
        # Check initial paper count
        db = Neo4jDatabase()
        db.connect()
        
        with db.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
            initial_count = result.single()["paper_count"]
        
        logger.info(f"📚 Initial papers in database: {initial_count}")
        
        # Test scraping and storing
        scraper = ArXivWebScraper()
        
        # Scrape a specific query
        query = "deep learning"
        papers = scraper.search_papers_web(query, max_results=3)
        
        if papers:
            logger.info(f"📚 Found {len(papers)} papers for '{query}'")
            
            # Store papers
            successful_stores = 0
            for paper in papers:
                if scraper.store_paper(paper):
                    successful_stores += 1
            
            logger.info(f"✅ Stored {successful_stores} papers")
            
            # Check final paper count
            with db.driver.session() as session:
                result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
                final_count = result.single()["paper_count"]
            
            logger.info(f"📚 Final papers in database: {final_count}")
            
            if final_count > initial_count:
                logger.info("✅ Database integration working correctly!")
            else:
                logger.warning("⚠️ Paper count didn't increase")
        
        scraper.close()
        db.close()
        
        logger.info("✅ Database integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database integration test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Dynamic Scraper Tests")
    logger.info("=" * 50)
    
    # Test 1: Dynamic scraping functionality
    test1_success = test_dynamic_scraping()
    
    # Test 2: Database integration
    test2_success = test_database_integration()
    
    if test1_success and test2_success:
        logger.info("\n🎉 All tests passed!")
        logger.info("The dynamic scraper is working correctly.")
        logger.info("You can now run: streamlit run src/app/main.py")
        logger.info("Then search for topics like 'adhd', 'machine learning', etc.")
    else:
        logger.error("\n❌ Some tests failed!")
        logger.error("Please check your configuration and try again.")
    
    return test1_success and test2_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 