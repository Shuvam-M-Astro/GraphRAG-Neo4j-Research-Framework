"""
Advanced GraphRAG Visualization Components
Provides interactive visualizations for graph-based research analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json

class AdvancedGraphVisualizer:
    def __init__(self):
        """Initialize the advanced graph visualizer."""
        self.color_scheme = {
            'Paper': '#1f77b4',
            'Author': '#ff7f0e', 
            'Method': '#2ca02c',
            'Keyword': '#d62728',
            'Citation': '#9467bd',
            'Collaboration': '#8c564b',
            'MethodUsage': '#e377c2'
        }
    
    def visualize_path_analysis(self, path_analysis: Dict[str, Any]) -> go.Figure:
        """
        Create interactive visualization for path analysis results.
        
        Args:
            path_analysis: Results from path analysis
            
        Returns:
            Plotly figure showing path analysis
        """
        if not path_analysis or "paths" not in path_analysis:
            return self._create_empty_figure("No path data available")
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes and edges from paths
        for path_data in path_analysis["paths"]:
            nodes = path_data["nodes"]
            relationships = path_data["relationships"]
            
            # Add nodes
            for i, node in enumerate(nodes):
                node_id = self._get_node_id(node)
                node_type = self._get_node_type(node)
                G.add_node(node_id, 
                          type=node_type,
                          label=self._get_node_label(node),
                          color=self.color_scheme.get(node_type, '#666666'))
            
            # Add edges
            for i, rel in enumerate(relationships):
                if i < len(nodes) - 1:
                    source = self._get_node_id(nodes[i])
                    target = self._get_node_id(nodes[i + 1])
                    G.add_edge(source, target, 
                              type=type(rel).__name__,
                              weight=path_data["strength"])
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[1]['label'])
            node_colors.append(node[1]['color'])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=f"Path Analysis: {path_analysis.get('source_entity', '')} → {path_analysis.get('target_entity', '')}",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def visualize_influence_propagation(self, propagation_data: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for influence propagation analysis.
        
        Args:
            propagation_data: Results from influence propagation analysis
            
        Returns:
            Plotly figure showing influence propagation
        """
        if not propagation_data or "propagation_paths" not in propagation_data:
            return self._create_empty_figure("No propagation data available")
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes and edges from propagation paths
        for path_data in propagation_data["propagation_paths"]:
            entity = path_data["entity"]
            score = path_data["score"]
            step = path_data["step"]
            
            node_id = self._get_node_id(entity)
            node_type = self._get_node_type(entity)
            G.add_node(node_id,
                      type=node_type,
                      label=self._get_node_label(entity),
                      score=score,
                      step=step,
                      color=self.color_scheme.get(node_type, '#666666'))
        
        # Add edges based on propagation steps
        for i, path_data in enumerate(propagation_data["propagation_paths"]):
            if i > 0:
                prev_entity = propagation_data["propagation_paths"][i-1]["entity"]
                curr_entity = path_data["entity"]
                
                prev_id = self._get_node_id(prev_entity)
                curr_id = self._get_node_id(curr_entity)
                
                if prev_id != curr_id:
                    G.add_edge(prev_id, curr_id, 
                              weight=path_data["score"],
                              step=path_data["step"])
        
        # Create layout with step-based positioning
        pos = {}
        for node, data in G.nodes(data=True):
            step = data.get('step', 0)
            pos[node] = (step, np.random.uniform(-1, 1))
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=edge[2]['weight'] * 5, color='#888'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node[1]['label']}<br>Score: {node[1]['score']:.3f}<br>Step: {node[1]['step']}")
            node_colors.append(node[1]['color'])
            node_sizes.append(node[1]['score'] * 30 + 10)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node[1]['label'] for node in G.nodes(data=True)],
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title="Influence Propagation Analysis",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(title="Propagation Step", showgrid=True),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def visualize_temporal_evolution(self, temporal_data: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for temporal evolution analysis.
        
        Args:
            temporal_data: Results from temporal analysis
            
        Returns:
            Plotly figure showing temporal evolution
        """
        if not temporal_data or "temporal_evolution" not in temporal_data:
            return self._create_empty_figure("No temporal data available")
        
        evolution = temporal_data["temporal_evolution"]
        
        # Prepare data for plotting
        years = []
        total_flows = []
        avg_flows = []
        flow_counts = []
        
        for year, metrics in evolution.items():
            years.append(year)
            total_flows.append(metrics.get('total_flow', 0))
            avg_flows.append(metrics.get('avg_flow', 0))
            flow_counts.append(metrics.get('flow_count', 0))
        
        # Create subplots
        fig = go.Figure()
        
        # Add total flow line
        fig.add_trace(go.Scatter(
            x=years, y=total_flows,
            mode='lines+markers',
            name='Total Knowledge Flow',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Add average flow line
        fig.add_trace(go.Scatter(
            x=years, y=avg_flows,
            mode='lines+markers',
            name='Average Flow Strength',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ))
        
        # Add flow count bars
        fig.add_trace(go.Bar(
            x=years, y=flow_counts,
            name='Number of Flows',
            marker_color='#2ca02c',
            opacity=0.7,
            yaxis='y3'
        ))
        
        # Update layout
        fig.update_layout(
            title="Temporal Evolution of Knowledge Flow",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Total Flow", side="left"),
            yaxis2=dict(title="Average Flow", side="right", overlaying="y"),
            yaxis3=dict(title="Flow Count", side="right", position=0.95),
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
    
    def visualize_cross_domain_analysis(self, cross_domain_data: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for cross-domain analysis.
        
        Args:
            cross_domain_data: Results from cross-domain analysis
            
        Returns:
            Plotly figure showing cross-domain connections
        """
        if not cross_domain_data or "bridging_concepts" not in cross_domain_data:
            return self._create_empty_figure("No cross-domain data available")
        
        bridging_concepts = cross_domain_data["bridging_concepts"]
        
        # Create sankey diagram for cross-domain flow
        if "concept_overlap" in bridging_concepts:
            # Prepare sankey data
            source = []
            target = []
            value = []
            labels = []
            
            for domain_pair, overlap in bridging_concepts["concept_overlap"].items():
                domains = domain_pair.split('_x_')
                if len(domains) == 2:
                    domain1, domain2 = domains
                    
                    # Add shared methods
                    if overlap["shared_methods"]:
                        source.append(domain1)
                        target.append(domain2)
                        value.append(len(overlap["shared_methods"]))
                        labels.extend([domain1, domain2])
                    
                    # Add shared keywords
                    if overlap["shared_keywords"]:
                        source.append(domain1)
                        target.append(domain2)
                        value.append(len(overlap["shared_keywords"]))
                        labels.extend([domain1, domain2])
            
            # Remove duplicates from labels
            unique_labels = list(dict.fromkeys(labels))
            
            # Create sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=unique_labels,
                    color="blue"
                ),
                link=dict(
                    source=[unique_labels.index(s) for s in source],
                    target=[unique_labels.index(t) for t in target],
                    value=value
                )
            )])
            
            fig.update_layout(
                title="Cross-Domain Knowledge Flow",
                font_size=10
            )
            
            return fig
        
        return self._create_empty_figure("No cross-domain connections found")
    
    def visualize_research_trends(self, trend_data: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for research trend analysis.
        
        Args:
            trend_data: Results from research trend analysis
            
        Returns:
            Plotly figure showing research trends
        """
        if not trend_data or "trend_analysis" not in trend_data:
            return self._create_empty_figure("No trend data available")
        
        trend_analysis = trend_data["trend_analysis"]
        
        # Create multiple subplots
        fig = go.Figure()
        
        # Add temporal results if available
        if "temporal_results" in trend_analysis:
            temporal_results = trend_analysis["temporal_results"]
            
            # Extract years and scores
            years = [paper.get("year", 0) for paper in temporal_results]
            scores = [paper.get("hybrid_score", 0) for paper in temporal_results]
            
            fig.add_trace(go.Scatter(
                x=years, y=scores,
                mode='markers',
                name='Research Relevance',
                marker=dict(
                    size=10,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
        
        # Add methodology evolution if available
        if "methodology_evolution" in trend_analysis:
            method_evolution = trend_analysis["methodology_evolution"]
            if "evolution" in method_evolution:
                evolution_data = method_evolution["evolution"]
                
                method_years = [item.get("year", 0) for item in evolution_data]
                method_counts = [len(item.get("keywords", [])) for item in evolution_data]
                
                fig.add_trace(go.Scatter(
                    x=method_years, y=method_counts,
                    mode='lines+markers',
                    name='Methodology Evolution',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
        
        fig.update_layout(
            title="Research Trends Analysis",
            xaxis=dict(title="Year"),
            yaxis=dict(title="Relevance Score", side="left"),
            yaxis2=dict(title="Methodology Count", side="right", overlaying="y"),
            hovermode='x unified'
        )
        
        return fig
    
    def _get_node_id(self, node: Dict) -> str:
        """Extract unique node ID from node data."""
        if "paper_id" in node:
            return f"paper_{node['paper_id']}"
        elif "name" in node:
            return f"entity_{node['name']}"
        elif "text" in node:
            return f"keyword_{node['text']}"
        else:
            return f"unknown_{id(node)}"
    
    def _get_node_type(self, node: Dict) -> str:
        """Extract node type from node data."""
        if "paper_id" in node:
            return "Paper"
        elif "name" in node and "institution" in node:
            return "Author"
        elif "name" in node and "description" in node:
            return "Method"
        elif "text" in node:
            return "Keyword"
        else:
            return "Unknown"
    
    def _get_node_label(self, node: Dict) -> str:
        """Extract display label from node data."""
        if "title" in node:
            return node["title"][:50] + "..." if len(node["title"]) > 50 else node["title"]
        elif "name" in node:
            return node["name"]
        elif "text" in node:
            return node["text"]
        else:
            return "Unknown"
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig 