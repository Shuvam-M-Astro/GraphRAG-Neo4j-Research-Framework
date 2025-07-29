"""
Streamlit Web Application for Graph RAG Scientific Research
Interactive interface for research analysis and exploration.
Enhanced with performance optimizations: lazy loading, caching, optimistic updates, and offline support.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os
import time
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag.orchestrator import GraphRAGOrchestrator
from app.performance import (
    performance_manager, lazy_loader, optimistic_updater, offline_manager,
    cached_result, show_performance_stats, show_offline_status, create_lazy_dataframe
)
from app.lazy_components import lazy_viz, lazy_table, lazy_search, lazy_metrics
from data_ingestion.arxiv_scraper import ArXivWebScraper

# Page configuration
st.set_page_config(
    page_title="Graph RAG Scientific Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .result-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def scrape_papers_for_query(query: str, max_results: int = 10):
    """Dynamically scrape papers from ArXiv for a specific query."""
    try:
        scraper = ArXivWebScraper()
        
        with st.spinner(f"🔍 Scraping ArXiv for '{query}'..."):
            # Scrape papers for the specific query (limited to 10)
            papers = scraper.search_papers_web(query, max_results)
            
            if papers:
                st.info(f"📚 Found {len(papers)} papers for '{query}' on ArXiv")
                
                # Store papers
                successful_stores = 0
                for paper in papers:
                    if scraper.store_paper(paper):
                        successful_stores += 1
                
                if successful_stores > 0:
                    st.success(f"✅ Successfully stored {successful_stores} papers for '{query}'!")
                    
                    # Create relationships for new papers
                    with st.spinner("Creating relationships..."):
                        scraper.create_collaboration_relationships()
                        scraper.create_citation_relationships()
                    
                    return successful_stores
                else:
                    st.warning("⚠️ No papers were stored successfully")
                    return 0
            else:
                st.warning(f"⚠️ No papers found for '{query}' on ArXiv")
                return 0
        
        scraper.close()
        
    except Exception as e:
        st.error(f"❌ Failed to scrape papers for '{query}': {e}")
        return 0

def check_and_populate_database():
    """Check if database has data and populate with sample data if needed."""
    try:
        from database.init_database import Neo4jDatabase
        
        db = Neo4jDatabase()
        db.connect()
        
        # Check if we have papers in the database
        with db.driver.session() as session:
            result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
            paper_count = result.single()["paper_count"]
        
        db.close()
        
        if paper_count == 0:
            st.info("📚 Database is empty. Loading sample data...")
            
            # Load sample data for initial setup
            db = Neo4jDatabase()
            db.connect()
            db.create_sample_data()
            db.close()
            st.success("✅ Sample data loaded successfully!")
        else:
            st.success(f"✅ Database has {paper_count} papers ready for analysis!")
            
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        st.info("💡 The system will continue with available data.")

def initialize_orchestrator():
    """Initialize the Graph RAG orchestrator."""
    try:
        with st.spinner("Initializing GraphRAG orchestrator..."):
            orchestrator = GraphRAGOrchestrator()
            st.success("✅ Orchestrator initialized successfully!")
            return orchestrator
    except Exception as e:
        st.error(f"❌ Failed to initialize orchestrator: {e}")
        
        # Provide helpful error messages for common issues
        if "meta tensor" in str(e).lower():
            st.info("💡 **Meta tensor error detected.** This is a known issue with newer PyTorch versions.")
            st.info("The system will attempt to use CPU-only initialization as a fallback.")
        elif "embedding model" in str(e).lower():
            st.info("💡 **Embedding model error detected.** This might be due to:")
            st.info("• Missing model files (check internet connection)")
            st.info("• PyTorch version compatibility issues")
            st.info("• Insufficient memory for model loading")
        elif "database" in str(e).lower():
            st.info("💡 **Database connection error detected.** Please check:")
            st.info("• Neo4j database is running")
            st.info("• Database credentials in .env file")
            st.info("• Network connectivity to database")
        
        return None

@cached_result(max_age_hours=2)
def display_papers_table(papers):
    """Display papers in a formatted table with lazy loading."""
    if not papers:
        st.warning("No papers found.")
        return
    
    # Check if we should use lazy loading for large datasets
    if len(papers) > 20:
        st.info(f"📊 Loading {len(papers)} papers with lazy loading...")
        create_lazy_dataframe(papers, page_size=10)
        return
    
    # Prepare data for display
    display_data = []
    for paper in papers:
        display_data.append({
            "Title": paper.get("title", "N/A"),
            "Year": paper.get("year", "N/A"),
            "Journal": paper.get("journal", "N/A"),
            "Citations": paper.get("citations", "N/A"),
            "Authors": ", ".join(paper.get("authors", [])),
            "Methods": ", ".join(paper.get("methods", [])),
            "Keywords": ", ".join(paper.get("keywords", []))
        })
    
    df = pd.DataFrame(display_data)
    st.dataframe(df, use_container_width=True)

def create_citation_network_plot(papers):
    """Create a citation network visualization."""
    if not papers:
        return None
    
    # Create nodes for papers
    nodes = []
    edges = []
    
    for i, paper in enumerate(papers):
        nodes.append({
            "id": paper.get("paper_id", f"paper_{i}"),
            "label": paper.get("title", "Unknown")[:50] + "...",
            "year": paper.get("year", 2020),
            "citations": paper.get("citations", 0)
        })
    
    # Create network visualization
    fig = go.Figure()
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in nodes:
        node_x.append(node["year"])
        node_y.append(node["citations"])
        node_text.append(f"{node['label']}<br>Year: {node['year']}<br>Citations: {node['citations']}")
        node_size.append(min(20 + node["citations"] * 0.5, 50))
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[node["label"] for node in nodes],
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_y,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Citations")
        ),
        hovertemplate="<b>%{text}</b><br>Year: %{x}<br>Citations: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Paper Citation Network",
        xaxis_title="Publication Year",
        yaxis_title="Number of Citations",
        height=500
    )
    
    return fig

def run_system_diagnostics():
    """Run system diagnostics to help troubleshoot issues."""
    st.subheader("🔧 System Diagnostics")
    
    # Check Python and package versions
    import sys
    import torch
    import transformers
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Python Version:**", sys.version)
        st.write("**PyTorch Version:**", torch.__version__)
        st.write("**Transformers Version:**", transformers.__version__)
    
    with col2:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        st.write("**CUDA Available:**", "✅ Yes" if cuda_available else "❌ No")
        
        if cuda_available:
            st.write("**CUDA Version:**", torch.version.cuda)
            st.write("**GPU Count:**", torch.cuda.device_count())
        
        # Check environment variables
        import os
        neo4j_uri = os.getenv("NEO4J_URI", "Not set")
        st.write("**Neo4j URI:**", neo4j_uri)
    
    # Test embedding model initialization
    if st.button("Test Embedding Model"):
        try:
            from graph_rag.graph_retriever import safe_initialize_sentence_transformer
            with st.spinner("Testing embedding model..."):
                model = safe_initialize_sentence_transformer("all-MiniLM-L6-v2")
                test_embedding = model.encode("Test sentence")
                st.success(f"✅ Embedding model test successful! Dimension: {len(test_embedding)}")
        except Exception as e:
            st.error(f"❌ Embedding model test failed: {e}")
    
    # Test database connection
    if st.button("Test Database Connection"):
        try:
            from database.init_database import Neo4jDatabase
            with st.spinner("Testing database connection..."):
                db = Neo4jDatabase()
                db.connect()
                with db.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                db.close()
                st.success(f"✅ Database connection successful! Test value: {test_value}")
        except Exception as e:
            st.error(f"❌ Database connection test failed: {e}")

def main():
    """Main application function."""
    
    # Check and populate database if needed
    check_and_populate_database()
    
    # Header with performance status
    st.markdown('<h1 class="main-header">🔬 GraphRAG Neo4j Research Framework</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Research Analysis with Graph-Based Retrieval and Generation")
    
    # Performance status bar
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if performance_manager.offline_mode:
            st.warning("🔄 Offline Mode")
        else:
            st.success("🌐 Online")
    with col2:
        cache_hit_rate = 0
        if sum(performance_manager.cache_stats.values()) > 0:
            cache_hit_rate = performance_manager.cache_stats["hits"] / sum(performance_manager.cache_stats.values()) * 100
        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
    with col3:
        st.metric("Cache Hits", performance_manager.cache_stats["hits"])
    with col4:
        st.metric("Cache Misses", performance_manager.cache_stats["misses"])
    with col5:
        # Show database status
        try:
            from database.init_database import Neo4jDatabase
            db = Neo4jDatabase()
            db.connect()
            with db.driver.session() as session:
                result = session.run("MATCH (p:Paper) RETURN count(p) as paper_count")
                paper_count = result.single()["paper_count"]
            db.close()
            st.metric("📚 Papers", paper_count)
        except:
            st.metric("📚 Papers", "N/A")
    
    # Initialize orchestrator
    orchestrator = initialize_orchestrator()
    if not orchestrator:
        st.error("Failed to initialize the system. Please check your configuration.")
        return
    
    # Sidebar for navigation and performance controls
    st.sidebar.title("Navigation")
    
    # Performance controls
    with st.sidebar.expander("⚡ Performance Settings", expanded=False):
        show_performance_stats()
        st.markdown("---")
        show_offline_status()
        st.markdown("---")
        
        # Cache management
        st.subheader("🗄️ Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Cache"):
                performance_manager.clear_cache()
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        # Database management
        st.subheader("🗄️ Database Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Database"):
                # Clear database
                try:
                    from database.init_database import Neo4jDatabase
                    db = Neo4jDatabase()
                    db.connect()
                    with db.driver.session() as session:
                        session.run("MATCH (n) DETACH DELETE n")
                    db.close()
                    st.success("✅ Database cleared successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to clear database: {e}")
        with col2:
            if st.button("📊 Manual Scrape"):
                # Manual scraping interface
                manual_query = st.text_input("Enter query to scrape:", key="manual_scrape_query")
                if manual_query:
                    scraped_count = scrape_papers_for_query(manual_query, max_results=10)
                    if scraped_count > 0:
                        st.success(f"✅ Successfully scraped {scraped_count} papers for '{manual_query}'!")
                    else:
                        st.warning(f"⚠️ No papers scraped for '{manual_query}'")
        
        # Cache info
        cache_files = len([f for f in os.listdir(performance_manager.cache_dir) if f.endswith('.pkl')])
        st.info(f"📁 Cached items: {cache_files}")
        
        # Offline data info
        offline_items = len(offline_manager.offline_data)
        st.info(f"📱 Offline items: {offline_items}")
        
        st.markdown("---")
        
        # Search history
        if lazy_search.search_history:
            st.subheader("🔍 Recent Searches")
            for i, search in enumerate(lazy_search.search_history[-5:]):  # Show last 5
                st.write(f"• {search['query']}")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "Research Topic Analysis",
            "Literature Review Generator",
            "Research Gap Analysis",
            "Methodology Evolution",
            "Collaboration Network",
            "Multi-hop Reasoning",
            "System Diagnostics"
        ]
    )
    
    # Main content area
    if page == "Research Topic Analysis":
        st.markdown('<h2 class="sub-header">📊 Research Topic Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Add search suggestions
            suggestions = lazy_search.get_search_suggestions("")
            if suggestions:
                st.info(f"💡 Recent searches: {', '.join(suggestions[:3])}")
            
            query = st.text_input("Enter your research topic or query:", 
                                 placeholder="e.g., deep learning in natural language processing")
            
            # Show suggestions as user types
            if query:
                query_suggestions = lazy_search.get_search_suggestions(query)
                if query_suggestions:
                    st.write("💡 Suggestions:")
                    for suggestion in query_suggestions:
                        if st.button(f"🔍 {suggestion}", key=f"sugg_{suggestion}"):
                            st.session_state['suggested_query'] = suggestion
                            st.rerun()
            
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["comprehensive", "gap", "methodology", "collaboration"]
            )
        
        if st.button("Analyze Research Topic", type="primary"):
            if query:
                # Use optimistic updates for immediate feedback
                start_time = time.time()
                
                # Show immediate feedback
                with st.container():
                    st.info(f"🔍 Analyzing research topic: {query}")
                    
                    # Create placeholders for results
                    metrics_placeholder = st.empty()
                    papers_placeholder = st.empty()
                    viz_placeholder = st.empty()
                    summary_placeholder = st.empty()
                    
                    # Show optimistic loading state
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Papers Found", "Loading...")
                        with col2:
                            st.metric("Avg Citations", "Loading...")
                        with col3:
                            st.metric("Avg Year", "Loading...")
                        with col4:
                            st.metric("Unique Methods", "Loading...")
                
                # First, scrape papers for the query
                scraped_count = scrape_papers_for_query(query, max_results=20)
                
                # Perform actual analysis
                try:
                    results = orchestrator.analyze_research_topic(query, analysis_type)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Calculate execution time
                        execution_time = time.time() - start_time
                        
                        # Display results with performance info
                        if scraped_count > 0:
                            st.success(f"✅ Analysis completed in {execution_time:.2f}s! Found {len(results['retrieved_papers'])} relevant papers (including {scraped_count} newly scraped from ArXiv).")
                        else:
                            st.success(f"✅ Analysis completed in {execution_time:.2f}s! Found {len(results['retrieved_papers'])} relevant papers.")
                        
                        # Update metrics with lazy loading
                        with metrics_placeholder.container():
                            lazy_metrics.display_lazy_metrics(results["retrieved_papers"])
                        
                        # Display papers with lazy loading
                        with papers_placeholder.container():
                            lazy_table.display_lazy_table(results["retrieved_papers"], "Retrieved Papers")
                        
                        # Lazy load visualization
                        if len(results["retrieved_papers"]) > 1:
                            with viz_placeholder.container():
                                st.subheader("📈 Citation Network")
                                # Use lazy loading for visualization
                                fig = lazy_viz.lazy_citation_network(results["retrieved_papers"])
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Display analysis results
                        with summary_placeholder.container():
                            if "research_summary" in results:
                                st.subheader("📋 Research Summary")
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.write(results["research_summary"])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if "gap_analysis" in results:
                                st.subheader("🔍 Gap Analysis")
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.write(results["gap_analysis"])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if "literature_review" in results:
                                st.subheader("📖 Literature Review")
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.write(results["literature_review"])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            if "research_recommendations" in results:
                                st.subheader("💡 Research Recommendations")
                                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                                st.write(results["research_recommendations"])
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Save to offline storage for offline access
                        offline_manager.save_offline_data(f"analysis_{query}", results)
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    # Show cached results if available
                    cached_results = offline_manager.get_offline_data(f"analysis_{query}")
                    if cached_results:
                        st.info("📱 Showing cached results (offline mode)")
                        # Display cached results
            else:
                st.warning("Please enter a research topic.")
    
    elif page == "Literature Review Generator":
        st.markdown('<h2 class="sub-header">📖 Literature Review Generator</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input("Enter research topic for literature review:", 
                                placeholder="e.g., transformer architecture in computer vision")
        
        with col2:
            max_papers = st.number_input("Max papers to analyze:", min_value=5, max_value=50, value=20)
        
        if st.button("Generate Literature Review", type="primary"):
            if topic:
                with st.spinner("Generating literature review..."):
                    results = orchestrator.generate_literature_review(topic, max_papers)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Literature review generated! Analyzed {results['papers_analyzed']} papers.")
                        
                        # Display papers
                        st.subheader("📚 Papers Analyzed")
                        display_papers_table(results["papers"])
                        
                        # Literature review
                        st.subheader("📖 Generated Literature Review")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["literature_review"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Research recommendations
                        st.subheader("💡 Research Recommendations")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["research_recommendations"])
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a research topic.")
    
    elif page == "Research Gap Analysis":
        st.markdown('<h2 class="sub-header">🔍 Research Gap Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            topic = st.text_input("Enter research topic for gap analysis:", 
                                placeholder="e.g., graph neural networks")
        
        with col2:
            max_hops = st.number_input("Max hops for analysis:", min_value=1, max_value=5, value=3)
        
        if st.button("Analyze Research Gaps", type="primary"):
            if topic:
                with st.spinner("Analyzing research gaps..."):
                    results = orchestrator.identify_research_gaps(topic, max_hops)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success("Research gap analysis completed!")
                        
                        # Gap analysis
                        st.subheader("🔍 Gap Analysis Results")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["gap_analysis"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Related papers
                        if "related_papers" in results:
                            st.subheader("📚 Related Papers")
                            display_papers_table(results["related_papers"])
            else:
                st.warning("Please enter a research topic.")
    
    elif page == "Methodology Evolution":
        st.markdown('<h2 class="sub-header">📈 Methodology Evolution Tracking</h2>', unsafe_allow_html=True)
        
        method_name = st.text_input("Enter methodology name to track:", 
                                  placeholder="e.g., Transformer")
        
        if st.button("Track Methodology Evolution", type="primary"):
            if method_name:
                with st.spinner("Tracking methodology evolution..."):
                    results = orchestrator.track_methodology_evolution(method_name)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Evolution analysis completed for {method_name}!")
                        
                        # Evolution report
                        st.subheader("📈 Evolution Report")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["evolution_report"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Evolution data visualization
                        if "evolution_data" in results and results["evolution_data"].get("evolution"):
                            st.subheader("📊 Evolution Timeline")
                            evolution_df = pd.DataFrame(results["evolution_data"]["evolution"])
                            if not evolution_df.empty:
                                fig = px.line(evolution_df, x="year", y="citations", 
                                            title=f"Evolution of {method_name} Over Time")
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a methodology name.")
    
    elif page == "Collaboration Network":
        st.markdown('<h2 class="sub-header">🤝 Collaboration Network Analysis</h2>', unsafe_allow_html=True)
        
        author_name = st.text_input("Enter author name (optional):", 
                                  placeholder="e.g., Dr. Sarah Johnson")
        
        if st.button("Analyze Collaboration Network", type="primary"):
            with st.spinner("Analyzing collaboration network..."):
                results = orchestrator.analyze_collaboration_network(author_name)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Collaboration analysis completed!")
                    
                    # Collaboration analysis
                    st.subheader("🤝 Collaboration Analysis")
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.write(results["collaboration_analysis"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Collaboration data
                    if "collaboration_data" in results and results["collaboration_data"].get("collaborations"):
                        st.subheader("📊 Collaboration Data")
                        collab_df = pd.DataFrame(results["collaboration_data"]["collaborations"])
                        st.dataframe(collab_df, use_container_width=True)
    
    elif page == "Multi-hop Reasoning":
        st.markdown('<h2 class="sub-header">🔄 Multi-hop Reasoning</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter research query for multi-hop reasoning:", 
                                placeholder="e.g., How do transformers influence computer vision?")
        
        with col2:
            max_hops = st.number_input("Max hops:", min_value=1, max_value=5, value=3)
        
        if st.button("Perform Multi-hop Reasoning", type="primary"):
            if query:
                with st.spinner("Performing multi-hop reasoning..."):
                    results = orchestrator.multi_hop_reasoning(query, max_hops)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"Multi-hop reasoning completed! Found {results['papers_found']} papers.")
                        
                        # Display papers
                        st.subheader("📚 Papers Found")
                        display_papers_table(results["papers"])
                        
                        # Research summary
                        st.subheader("📋 Research Summary")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["research_summary"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Research recommendations
                        st.subheader("💡 Research Recommendations")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.write(results["research_recommendations"])
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a research query.")
    
    elif page == "System Diagnostics":
        run_system_diagnostics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 GraphRAG Neo4j Research Framework | Powered by Neo4j, OpenAI, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 