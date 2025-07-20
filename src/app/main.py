"""
Streamlit Web Application for Graph RAG Scientific Research
Interactive interface for research analysis and exploration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag.orchestrator import GraphRAGOrchestrator

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

@st.cache_resource
def initialize_orchestrator():
    """Initialize the Graph RAG orchestrator with caching."""
    try:
        return GraphRAGOrchestrator()
    except Exception as e:
        st.error(f"Failed to initialize orchestrator: {e}")
        return None

def display_papers_table(papers):
    """Display papers in a formatted table."""
    if not papers:
        st.warning("No papers found.")
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

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🔬 GraphRAG Neo4j Research Framework</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Research Analysis with Graph-Based Retrieval and Generation")
    
    # Initialize orchestrator
    orchestrator = initialize_orchestrator()
    if not orchestrator:
        st.error("Failed to initialize the system. Please check your configuration.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "Research Topic Analysis",
            "Literature Review Generator",
            "Research Gap Analysis",
            "Methodology Evolution",
            "Collaboration Network",
            "Multi-hop Reasoning"
        ]
    )
    
    # Main content area
    if page == "Research Topic Analysis":
        st.markdown('<h2 class="sub-header">📊 Research Topic Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query = st.text_input("Enter your research topic or query:", 
                                 placeholder="e.g., deep learning in natural language processing")
            
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["comprehensive", "gap", "methodology", "collaboration"]
            )
        
        if st.button("Analyze Research Topic", type="primary"):
            if query:
                with st.spinner("Analyzing research topic..."):
                    results = orchestrator.analyze_research_topic(query, analysis_type)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Display results
                        st.success(f"Analysis completed! Found {len(results['retrieved_papers'])} relevant papers.")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Papers Found", len(results["retrieved_papers"]))
                        with col2:
                            avg_citations = sum(p.get("citations", 0) for p in results["retrieved_papers"]) / len(results["retrieved_papers"])
                            st.metric("Avg Citations", f"{avg_citations:.1f}")
                        with col3:
                            avg_year = sum(p.get("year", 2020) for p in results["retrieved_papers"]) / len(results["retrieved_papers"])
                            st.metric("Avg Year", f"{avg_year:.0f}")
                        with col4:
                            unique_methods = len(set(method for p in results["retrieved_papers"] for method in p.get("methods", [])))
                            st.metric("Unique Methods", unique_methods)
                        
                        # Display papers
                        st.subheader("📚 Retrieved Papers")
                        display_papers_table(results["retrieved_papers"])
                        
                        # Citation network
                        if len(results["retrieved_papers"]) > 1:
                            st.subheader("📈 Citation Network")
                            fig = create_citation_network_plot(results["retrieved_papers"])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Analysis results
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 GraphRAG Neo4j Research Framework | Powered by Neo4j, OpenAI, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 