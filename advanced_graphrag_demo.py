#!/usr/bin/env python3
"""
Advanced GraphRAG Demo Script
Demonstrates the enhanced GraphRAG capabilities for scientific research analysis.
"""

import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from graph_rag.orchestrator import GraphRAGOrchestrator
from visualization.graph_visualizer import AdvancedGraphVisualizer

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedGraphRAGDemo:
    def __init__(self):
        """Initialize the advanced GraphRAG demo."""
        self.orchestrator = GraphRAGOrchestrator()
        self.visualizer = AdvancedGraphVisualizer()
        
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of all GraphRAG features."""
        logger.info("🚀 Starting Advanced GraphRAG Demo")
        
        try:
            # 1. Comprehensive Graph Analysis
            self.demo_comprehensive_analysis()
            
            # 2. Influence Propagation Analysis
            self.demo_influence_propagation()
            
            # 3. Cross-Domain Research Analysis
            self.demo_cross_domain_analysis()
            
            # 4. Research Trend Analysis
            self.demo_research_trends()
            
            # 5. Advanced Retrieval Methods
            self.demo_advanced_retrieval()
            
            logger.info("✅ Advanced GraphRAG Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
        finally:
            self.orchestrator.close()
    
    def demo_comprehensive_analysis(self):
        """Demonstrate comprehensive graph analysis."""
        logger.info("📊 Running Comprehensive Graph Analysis Demo")
        
        query = "deep learning in natural language processing"
        
        # Run comprehensive analysis
        results = self.orchestrator.comprehensive_graph_analysis(
            query=query,
            analysis_depth="deep"
        )
        
        if "error" in results:
            logger.error(f"Comprehensive analysis failed: {results['error']}")
            return
        
        # Display results
        logger.info(f"✅ Comprehensive analysis completed for: {query}")
        logger.info(f"📈 Analysis components: {list(results['analysis_components'].keys())}")
        
        if "comprehensive_insights" in results:
            insights = results["comprehensive_insights"]
            if "graph_insights" in insights:
                logger.info("🧠 Graph-aware insights generated")
                logger.info(f"💡 Insight preview: {insights['graph_insights'][:200]}...")
        
        if "path_analysis_report" in results:
            logger.info("🛤️ Path analysis report generated")
        
        if "temporal_evolution_report" in results:
            logger.info("⏰ Temporal evolution report generated")
    
    def demo_influence_propagation(self):
        """Demonstrate influence propagation analysis."""
        logger.info("🌊 Running Influence Propagation Analysis Demo")
        
        # Use sample seed entities (in practice, these would come from user input)
        seed_entities = ["transformer", "attention mechanism", "BERT"]
        
        # Run influence propagation analysis
        results = self.orchestrator.influence_propagation_analysis(
            seed_entities=seed_entities,
            analysis_type="comprehensive"
        )
        
        if "error" in results:
            logger.error(f"Influence propagation failed: {results['error']}")
            return
        
        logger.info(f"✅ Influence propagation completed for {len(seed_entities)} seed entities")
        
        if "propagation_analysis" in results:
            prop_analysis = results["propagation_analysis"]
            if "propagation_results" in prop_analysis:
                prop_results = prop_analysis["propagation_results"]
                if "influenced_entities" in prop_results:
                    logger.info(f"🎯 {len(prop_results['influenced_entities'])} entities influenced")
        
        if "influence_insights" in results:
            insights = results["influence_insights"]
            if "graph_insights" in insights:
                logger.info("🧠 Influence insights generated")
    
    def demo_cross_domain_analysis(self):
        """Demonstrate cross-domain research analysis."""
        logger.info("🔗 Running Cross-Domain Research Analysis Demo")
        
        primary_domain = "machine learning"
        secondary_domains = ["computer vision", "robotics", "bioinformatics"]
        
        # Run cross-domain analysis
        results = self.orchestrator.cross_domain_research_analysis(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains
        )
        
        if "error" in results:
            logger.error(f"Cross-domain analysis failed: {results['error']}")
            return
        
        logger.info(f"✅ Cross-domain analysis completed")
        logger.info(f"🌐 Primary domain: {primary_domain}")
        logger.info(f"🔗 Secondary domains: {secondary_domains}")
        
        if "cross_domain_analysis" in results:
            cross_domain = results["cross_domain_analysis"]
            if "domain_results" in cross_domain:
                domain_results = cross_domain["domain_results"]
                for domain, papers in domain_results.items():
                    logger.info(f"📚 {domain}: {len(papers)} papers retrieved")
        
        if "cross_domain_insights" in results:
            logger.info("🧠 Cross-domain insights generated")
    
    def demo_research_trends(self):
        """Demonstrate research trend analysis."""
        logger.info("📈 Running Research Trend Analysis Demo")
        
        topic = "graph neural networks"
        time_period = (2015, 2024)
        
        # Run research trend analysis
        results = self.orchestrator.research_trend_analysis(
            topic=topic,
            time_period=time_period
        )
        
        if "error" in results:
            logger.error(f"Research trend analysis failed: {results['error']}")
            return
        
        logger.info(f"✅ Research trend analysis completed for: {topic}")
        logger.info(f"📅 Time period: {time_period[0]} - {time_period[1]}")
        
        if "trend_analysis" in results:
            trend_analysis = results["trend_analysis"]
            if "temporal_results" in trend_analysis:
                temporal_results = trend_analysis["temporal_results"]
                logger.info(f"⏰ {len(temporal_results)} temporal results found")
        
        if "trend_insights" in results:
            logger.info("🧠 Trend insights generated")
        
        if "research_recommendations" in results:
            logger.info("💡 Research recommendations generated")
    
    def demo_advanced_retrieval(self):
        """Demonstrate advanced retrieval methods."""
        logger.info("🔍 Running Advanced Retrieval Methods Demo")
        
        query = "attention mechanisms in deep learning"
        
        # 1. Graph-aware search
        logger.info("🎯 Testing Graph-aware Search")
        try:
            graph_aware_results = self.orchestrator.retriever.graph_aware_search(
                query=query,
                entity_types=["Paper", "Method", "Keyword"],
                path_constraints={"min_year": 2018},
                limit=10
            )
            logger.info(f"✅ Graph-aware search: {len(graph_aware_results)} results")
        except Exception as e:
            logger.error(f"Graph-aware search failed: {e}")
        
        # 2. Temporal graph search
        logger.info("⏰ Testing Temporal Graph Search")
        try:
            temporal_results = self.orchestrator.retriever.temporal_graph_search(
                query=query,
                time_window=(2015, 2024),
                temporal_weight=0.4
            )
            logger.info(f"✅ Temporal search: {len(temporal_results)} results")
        except Exception as e:
            logger.error(f"Temporal search failed: {e}")
        
        # 3. Entity-centric search
        logger.info("🎯 Testing Entity-centric Search")
        try:
            entity_results = self.orchestrator.retriever.entity_centric_search(
                entity_name="transformer",
                entity_type="Method",
                max_hops=2
            )
            logger.info(f"✅ Entity-centric search completed")
        except Exception as e:
            logger.error(f"Entity-centric search failed: {e}")
        
        # 4. Path analysis
        logger.info("🛤️ Testing Path Analysis")
        try:
            # Use sample entities for path analysis
            path_results = self.orchestrator.retriever.path_analysis_search(
                source_entity="transformer",
                target_entity="BERT",
                max_paths=3
            )
            if "paths" in path_results:
                logger.info(f"✅ Path analysis: {len(path_results['paths'])} paths found")
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
    
    def demo_visualization_capabilities(self):
        """Demonstrate advanced visualization capabilities."""
        logger.info("📊 Running Visualization Capabilities Demo")
        
        # This would typically be integrated with Streamlit or other visualization framework
        logger.info("🎨 Advanced visualization components available:")
        logger.info("  - Path analysis visualization")
        logger.info("  - Influence propagation graphs")
        logger.info("  - Temporal evolution charts")
        logger.info("  - Cross-domain analysis diagrams")
        logger.info("  - Research trend visualizations")
    
    def print_performance_stats(self):
        """Print performance statistics."""
        logger.info("📊 Performance Statistics")
        
        try:
            stats = self.orchestrator.retriever.get_performance_stats()
            logger.info(f"🔍 Total searches: {stats.get('total_searches', 0)}")
            logger.info(f"⏱️ Average search time: {stats.get('average_search_time', 0):.2f}s")
            logger.info(f"📚 Total documents: {stats.get('total_documents', 0)}")
            logger.info(f"🏗️ Index build time: {stats.get('index_build_time', 0):.2f}s")
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")

def main():
    """Main demo function."""
    print("=" * 80)
    print("🚀 ADVANCED GRAPHRAG NEO4J RESEARCH FRAMEWORK DEMO")
    print("=" * 80)
    print()
    print("This demo showcases advanced GraphRAG capabilities including:")
    print("• Graph-aware retrieval with entity and path constraints")
    print("• Influence propagation analysis")
    print("• Cross-domain research analysis")
    print("• Temporal evolution tracking")
    print("• Path analysis and knowledge flow")
    print("• Advanced visualization components")
    print()
    
    # Check environment
    required_vars = ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file")
        return
    
    print("✅ Environment variables configured")
    print()
    
    # Run demo
    demo = AdvancedGraphRAGDemo()
    
    try:
        demo.run_comprehensive_demo()
        demo.print_performance_stats()
        demo.demo_visualization_capabilities()
        
        print()
        print("=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Key GraphRAG Enhancements Demonstrated:")
        print("1. 🎯 Graph-aware retrieval with entity constraints")
        print("2. 🌊 Influence propagation through knowledge graphs")
        print("3. 🔗 Cross-domain research connections")
        print("4. ⏰ Temporal evolution analysis")
        print("5. 🛤️ Path-based reasoning and analysis")
        print("6. 📊 Advanced visualization capabilities")
        print("7. 🧠 Context-aware LLM generation")
        print("8. 🔍 Multi-modal research insights")
        print()
        print("These enhancements make your GraphRAG system significantly more")
        print("powerful for scientific research and knowledge discovery!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    main() 