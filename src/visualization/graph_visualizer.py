"""
Graph Visualization for Scientific Research Knowledge Graph
Creates interactive visualizations of research networks and relationships.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json

load_dotenv()

logger = logging.getLogger(__name__)

class GraphVisualizer:
    def __init__(self):
        """Initialize graph visualizer with Neo4j connection."""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        self.connect()

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

    def get_collaboration_network(self, author_name: str = None, max_authors: int = 50) -> Dict[str, Any]:
        """
        Get collaboration network data.
        
        Args:
            author_name: Specific author to focus on (optional)
            max_authors: Maximum number of authors to include
            
        Returns:
            Network data for visualization
        """
        try:
            with self.driver.session() as session:
                if author_name:
                    # Get collaboration network for specific author
                    result = session.run("""
                        MATCH (a:Author {name: $author_name})-[:COLLABORATED_WITH]-(collaborator:Author)
                        RETURN a.name as author1, collaborator.name as author2
                        LIMIT $max_authors
                    """, {"author_name": author_name, "max_authors": max_authors})
                else:
                    # Get overall collaboration network
                    result = session.run("""
                        MATCH (a1:Author)-[:COLLABORATED_WITH]-(a2:Author)
                        RETURN a1.name as author1, a2.name as author2
                        LIMIT $max_authors
                    """, {"max_authors": max_authors})
                
                edges = []
                nodes = set()
                
                for record in result:
                    author1 = record["author1"]
                    author2 = record["author2"]
                    edges.append({"source": author1, "target": author2})
                    nodes.add(author1)
                    nodes.add(author2)
                
                return {
                    "nodes": [{"id": node, "label": node} for node in nodes],
                    "edges": edges
                }
                
        except Exception as e:
            logger.error(f"Error getting collaboration network: {e}")
            return {"nodes": [], "edges": []}

    def create_collaboration_network_plot(self, network_data: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive collaboration network visualization.
        
        Args:
            network_data: Network data from get_collaboration_network
            
        Returns:
            Plotly figure object
        """
        if not network_data["nodes"]:
            return go.Figure()
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in network_data["nodes"]:
            G.add_node(node["id"])
        
        # Add edges
        for edge in network_data["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node traces
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=20,
                color=[],
                line_width=2))
        
        # Color nodes by degree
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Collaboration Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig

    def get_methodology_evolution_data(self, method_name: str) -> Dict[str, Any]:
        """
        Get methodology evolution data over time.
        
        Args:
            method_name: Name of the methodology
            
        Returns:
            Evolution data for visualization
        """
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: $method_name})
                    RETURN p.year as year, count(p) as paper_count
                    ORDER BY p.year
                """, {"method_name": method_name})
                
                years = []
                counts = []
                
                for record in result:
                    years.append(record["year"])
                    counts.append(record["paper_count"])
                
                return {
                    "method": method_name,
                    "years": years,
                    "counts": counts
                }
                
        except Exception as e:
            logger.error(f"Error getting methodology evolution data: {e}")
            return {"method": method_name, "years": [], "counts": []}

    def create_methodology_evolution_plot(self, evolution_data: Dict[str, Any]) -> go.Figure:
        """
        Create methodology evolution timeline plot.
        
        Args:
            evolution_data: Evolution data from get_methodology_evolution_data
            
        Returns:
            Plotly figure object
        """
        if not evolution_data["years"]:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=evolution_data["years"],
            y=evolution_data["counts"],
            mode='lines+markers',
            name=evolution_data["method"],
            line=dict(width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Evolution of {evolution_data["method"]} Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Papers',
            height=400
        )
        
        return fig

    def get_research_trends_data(self, topic: str = None) -> Dict[str, Any]:
        """
        Get research trends data.
        
        Args:
            topic: Optional topic filter
            
        Returns:
            Trends data for visualization
        """
        try:
            with self.driver.session() as session:
                if topic:
                    # Get trends for specific topic
                    result = session.run("""
                        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                        WHERE toLower(k.text) CONTAINS toLower($topic)
                        RETURN p.year as year, count(p) as paper_count
                        ORDER BY p.year
                    """, {"topic": topic})
                else:
                    # Get overall trends
                    result = session.run("""
                        MATCH (p:Paper)
                        RETURN p.year as year, count(p) as paper_count
                        ORDER BY p.year
                    """)
                
                years = []
                counts = []
                
                for record in result:
                    years.append(record["year"])
                    counts.append(record["paper_count"])
                
                return {
                    "topic": topic or "All Research",
                    "years": years,
                    "counts": counts
                }
                
        except Exception as e:
            logger.error(f"Error getting research trends data: {e}")
            return {"topic": topic or "All Research", "years": [], "counts": []}

    def create_research_trends_plot(self, trends_data: Dict[str, Any]) -> go.Figure:
        """
        Create research trends visualization.
        
        Args:
            trends_data: Trends data from get_research_trends_data
            
        Returns:
            Plotly figure object
        """
        if not trends_data["years"]:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trends_data["years"],
            y=trends_data["counts"],
            mode='lines+markers',
            name=trends_data["topic"],
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f'Research Trends: {trends_data["topic"]}',
            xaxis_title='Year',
            yaxis_title='Number of Papers',
            height=400
        )
        
        return fig

    def get_citation_network_data(self, paper_id: str = None, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get citation network data.
        
        Args:
            paper_id: Specific paper to focus on (optional)
            max_depth: Maximum depth for citation traversal
            
        Returns:
            Citation network data
        """
        try:
            with self.driver.session() as session:
                if paper_id:
                    # Get citation network for specific paper (without APOC)
                    query = f"""
                        MATCH (start:Paper {{paper_id: $paper_id}})
                        WITH start
                        OPTIONAL MATCH (start)-[:CITES*1..{max_depth}]-(related:Paper)
                        RETURN related.paper_id as paper_id, related.title as title, 
                               related.year as year, related.citations as citations
                    """
                    result = session.run(query, {"paper_id": paper_id})
                else:
                    # Get overall citation network
                    result = session.run("""
                        MATCH (p:Paper)
                        RETURN p.paper_id as paper_id, p.title as title, 
                               p.year as year, p.citations as citations
                        ORDER BY p.citations DESC
                        LIMIT 50
                    """)
                
                papers = []
                for record in result:
                    papers.append({
                        "paper_id": record["paper_id"],
                        "title": record["title"],
                        "year": record["year"],
                        "citations": record["citations"] or 0
                    })
                
                return {"papers": papers}
                
        except Exception as e:
            logger.error(f"Error getting citation network data: {e}")
            return {"papers": []}

    def create_citation_network_plot(self, citation_data: Dict[str, Any]) -> go.Figure:
        """
        Create citation network visualization.
        
        Args:
            citation_data: Citation data from get_citation_network_data
            
        Returns:
            Plotly figure object
        """
        if not citation_data["papers"]:
            return go.Figure()
        
        papers = citation_data["papers"]
        
        # Create scatter plot
        fig = go.Figure()
        
        x = [paper["year"] for paper in papers]
        y = [paper["citations"] for paper in papers]
        text = [paper["title"][:50] + "..." for paper in papers]
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            text=text,
            textposition="top center",
            marker=dict(
                size=[min(20 + paper["citations"] * 0.5, 50) for paper in papers],
                color=y,
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

    def get_keyword_cloud_data(self, topic: str = None) -> Dict[str, Any]:
        """
        Get keyword frequency data for word cloud.
        
        Args:
            topic: Optional topic filter
            
        Returns:
            Keyword frequency data
        """
        try:
            with self.driver.session() as session:
                if topic:
                    # Get keywords for specific topic
                    result = session.run("""
                        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                        WHERE toLower(k.text) CONTAINS toLower($topic)
                        RETURN k.text as keyword, count(p) as frequency
                        ORDER BY frequency DESC
                        LIMIT 50
                    """, {"topic": topic})
                else:
                    # Get overall keyword frequencies
                    result = session.run("""
                        MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
                        RETURN k.text as keyword, count(p) as frequency
                        ORDER BY frequency DESC
                        LIMIT 50
                    """)
                
                keywords = []
                frequencies = []
                
                for record in result:
                    keywords.append(record["keyword"])
                    frequencies.append(record["frequency"])
                
                return {
                    "keywords": keywords,
                    "frequencies": frequencies
                }
                
        except Exception as e:
            logger.error(f"Error getting keyword cloud data: {e}")
            return {"keywords": [], "frequencies": []}

    def create_keyword_cloud_plot(self, keyword_data: Dict[str, Any]) -> go.Figure:
        """
        Create keyword frequency visualization.
        
        Args:
            keyword_data: Keyword data from get_keyword_cloud_data
            
        Returns:
            Plotly figure object
        """
        if not keyword_data["keywords"]:
            return go.Figure()
        
        # Create bar chart for keyword frequencies
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=keyword_data["keywords"],
            y=keyword_data["frequencies"],
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title="Keyword Frequency Distribution",
            xaxis_title="Keywords",
            yaxis_title="Frequency",
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig 