#!/usr/bin/env python3
"""
Visualize Neo4j graph using NetworkX and Matplotlib
"""

import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

load_dotenv()

def visualize_graph():
    """Visualize the Neo4j graph."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Get all nodes and relationships
            result = session.run("""
                MATCH (n)-[r]->(m)
                RETURN n, r, m
            """)
            
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Node colors and labels
            node_colors = []
            node_labels = {}
            
            for record in result:
                source_node = record["n"]
                target_node = record["m"]
                relationship = record["r"]
                
                # Add nodes
                source_id = f"{list(source_node.labels)[0]}_{source_node.get('paper_id', source_node.get('author_id', source_node.get('method_id', source_node.get('keyword_id', 'unknown'))))}"
                target_id = f"{list(target_node.labels)[0]}_{target_node.get('paper_id', target_node.get('author_id', target_node.get('method_id', target_node.get('keyword_id', 'unknown'))))}"
                
                # Add source node
                if source_id not in G.nodes():
                    G.add_node(source_id)
                    node_type = list(source_node.labels)[0]
                    if node_type == "Paper":
                        node_colors.append('lightblue')
                        node_labels[source_id] = source_node.get('title', '')[:20] + "..."
                    elif node_type == "Author":
                        node_colors.append('lightgreen')
                        node_labels[source_id] = source_node.get('name', '')
                    elif node_type == "Method":
                        node_colors.append('orange')
                        node_labels[source_id] = source_node.get('name', '')
                    elif node_type == "Keyword":
                        node_colors.append('pink')
                        node_labels[source_id] = source_node.get('text', '')
                
                # Add target node
                if target_id not in G.nodes():
                    G.add_node(target_id)
                    node_type = list(target_node.labels)[0]
                    if node_type == "Paper":
                        node_colors.append('lightblue')
                        node_labels[target_id] = target_node.get('title', '')[:20] + "..."
                    elif node_type == "Author":
                        node_colors.append('lightgreen')
                        node_labels[target_id] = target_node.get('name', '')
                    elif node_type == "Method":
                        node_colors.append('orange')
                        node_labels[target_id] = target_node.get('name', '')
                    elif node_type == "Keyword":
                        node_colors.append('pink')
                        node_labels[target_id] = target_node.get('text', '')
                
                # Add edge
                G.add_edge(source_id, target_id, label=type(relationship).__name__)
            
            # Create the visualization
            plt.figure(figsize=(15, 10))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='lightblue', label='Papers'),
                mpatches.Patch(color='lightgreen', label='Authors'),
                mpatches.Patch(color='orange', label='Methods'),
                mpatches.Patch(color='pink', label='Keywords')
            ]
            plt.legend(handles=legend_elements, loc='upper left')
            
            plt.title("GraphRAG Research Knowledge Graph", fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('research_graph.png', dpi=300, bbox_inches='tight')
            print("📊 Graph visualization saved as 'research_graph.png'")
            
            # Show the plot
            plt.show()
            
            # Print graph statistics
            print(f"\n📈 Graph Statistics:")
            print(f"  Nodes: {G.number_of_nodes()}")
            print(f"  Edges: {G.number_of_edges()}")
            print(f"  Node types: {set([node.split('_')[0] for node in G.nodes()])}")
            
    except Exception as e:
        print(f"❌ Error visualizing graph: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    visualize_graph() 