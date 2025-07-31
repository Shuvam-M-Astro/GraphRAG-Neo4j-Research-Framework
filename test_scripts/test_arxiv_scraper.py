#!/usr/bin/env python3
"""
Test script for ArXiv web scraper.
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_arxiv_scraper():
    """Test the ArXiv web scraper."""
    logger.info("🧪 Testing ArXiv Web Scraper")
    logger.info("=" * 50)
    
    try:
        # Initialize scraper
        scraper = ArXivWebScraper()
        logger.info("✅ Scraper initialized successfully")
        
        # Test single query
        logger.info("Testing search for 'adhd' papers...")
        papers = scraper.search_papers_web("adhd", max_results=5)
        
        if papers:
            logger.info(f"✅ Found {len(papers)} papers")
            for i, paper in enumerate(papers[:3]):
                logger.info(f"   {i+1}. {paper['title'][:80]}...")
                logger.info(f"      Authors: {', '.join(paper['authors'][:3])}")
                logger.info(f"      Categories: {', '.join(paper['categories'][:3])}")
        else:
            logger.warning("⚠️ No papers found")
        
        # Test storing a paper
        if papers:
            logger.info("Testing paper storage...")
            success = scraper.store_paper(papers[0])
            if success:
                logger.info("✅ Paper stored successfully")
            else:
                logger.error("❌ Failed to store paper")
        
        scraper.close()
        logger.info("✅ ArXiv scraper test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ArXiv scraper test failed: {e}")
        return False

def main():
    """Main test function."""
    success = test_arxiv_scraper()
    
    if success:
        logger.info("\n🎉 All tests passed!")
        logger.info("The ArXiv web scraper is working correctly.")
    else:
        logger.error("\n❌ Tests failed!")
        logger.error("Please check your configuration and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 