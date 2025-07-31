#!/usr/bin/env python3
"""
Run ArXiv web scraper to populate the database with real data.
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

def main():
    """Main function to run ArXiv web scraper."""
    logger.info("🚀 Starting ArXiv Web Scraper")
    logger.info("=" * 50)
    
    # Define queries to scrape
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
        logger.info(f"Scraping {len(queries)} queries from ArXiv...")
        
        # Scrape and store papers
        stats = scraper.scrape_and_store_papers(queries, max_results_per_query=10)
        
        logger.info("=" * 50)
        logger.info("📊 Scraping Results:")
        logger.info(f"   Total papers found: {stats['total_papers_found']}")
        logger.info(f"   Successfully stored: {stats['successful_stores']}")
        logger.info(f"   Failed stores: {stats['failed_stores']}")
        logger.info(f"   Success rate: {stats['success_rate']:.2%}")
        logger.info(f"   Pages scraped: {stats['pages_scraped']}")
        logger.info(f"   Total processing time: {stats['total_processing_time']:.2f}s")
        
        # Create relationships
        logger.info("Creating collaboration relationships...")
        scraper.create_collaboration_relationships()
        
        logger.info("Creating citation relationships...")
        scraper.create_citation_relationships()
        
        logger.info("=" * 50)
        logger.info("🎉 ArXiv web scraping completed successfully!")
        logger.info("")
        logger.info("You can now search for topics like:")
        logger.info("• 'adhd' - ADHD research papers")
        logger.info("• 'deep learning' - Machine learning papers")
        logger.info("• 'transformer' - Transformer architecture papers")
        logger.info("• 'computer vision' - Computer vision papers")
        logger.info("")
        logger.info("Try running the web app: streamlit run src/app/main.py")
        
    except Exception as e:
        logger.error(f"❌ ArXiv web scraping failed: {e}")
        return False
    finally:
        scraper.close()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 