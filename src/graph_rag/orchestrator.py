"""
Graph RAG Orchestrator for Scientific Research
Coordinates retrieval and generation for comprehensive research analysis.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .graph_retriever import GraphRetriever
from .generator import GraphRAGGenerator

load_dotenv()

logger = logging.getLogger(__name__)

class GraphRAGOrchestrator:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gpt-4"):
        """
        Initialize the Graph RAG orchestrator.
        
        Args:
            embedding_model: Sentence transformer model for embeddings
            llm_model: OpenAI model for generation
        """
        self.retriever = GraphRetriever(model_name=embedding_model)
        self.generator = GraphRAGGenerator(model_name=llm_model)
        
    def analyze_research_topic(self, query: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive research topic analysis.
        
        Args:
            query: Research query or topic
            analysis_type: Type of analysis ("comprehensive", "gap", "methodology", "collaboration")
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Retrieve relevant papers using hybrid search
            retrieved_papers = self.retriever.hybrid_search(query, alpha=0.7, limit=15)
            
            if not retrieved_papers:
                return {"error": "No relevant papers found for the query"}
            
            results = {
                "query": query,
                "retrieved_papers": retrieved_papers,
                "analysis_type": analysis_type
            }
            
            if analysis_type == "comprehensive":
                # Generate comprehensive analysis
                results["research_summary"] = self.generator.generate_research_summary(retrieved_papers, query)
                results["literature_review"] = self.generator.generate_literature_review(retrieved_papers, query)
                results["research_recommendations"] = self.generator.generate_research_recommendations(retrieved_papers, query)
                
                # Get research gaps
                gaps_data = self.retriever.get_research_gaps(query)
                if "error" not in gaps_data:
                    results["gap_analysis"] = self.generator.generate_gap_analysis(gaps_data)
                    results["gap_data"] = gaps_data
                
            elif analysis_type == "gap":
                # Focus on gap analysis
                gaps_data = self.retriever.get_research_gaps(query)
                if "error" not in gaps_data:
                    results["gap_analysis"] = self.generator.generate_gap_analysis(gaps_data)
                    results["gap_data"] = gaps_data
                else:
                    results["error"] = gaps_data["error"]
                    
            elif analysis_type == "methodology":
                # Focus on methodology analysis
                # Extract methodology names from retrieved papers
                methodologies = set()
                for paper in retrieved_papers:
                    if paper.get("methods"):
                        methodologies.update(paper["methods"])
                
                methodology_analyses = {}
                for method in list(methodologies)[:3]:  # Limit to top 3 methods
                    evolution_data = self.retriever.get_methodology_evolution(method)
                    methodology_analyses[method] = {
                        "evolution_report": self.generator.generate_methodology_evolution_report(evolution_data),
                        "evolution_data": evolution_data
                    }
                
                results["methodology_analyses"] = methodology_analyses
                
            elif analysis_type == "collaboration":
                # Focus on collaboration analysis
                collaboration_data = self.retriever.get_collaboration_network()
                results["collaboration_analysis"] = self.generator.generate_collaboration_analysis(collaboration_data)
                results["collaboration_data"] = collaboration_data
            
            return results
            
        except Exception as e:
            logger.error(f"Error in research topic analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def generate_literature_review(self, topic: str, max_papers: int = 20) -> Dict[str, Any]:
        """
        Generate a comprehensive literature review for a research topic.
        
        Args:
            topic: Research topic
            max_papers: Maximum number of papers to include
            
        Returns:
            Literature review results
        """
        try:
            # Retrieve papers using hybrid search
            retrieved_papers = self.retriever.hybrid_search(topic, alpha=0.6, limit=max_papers)
            
            if not retrieved_papers:
                return {"error": "No relevant papers found for the topic"}
            
            # Generate literature review
            literature_review = self.generator.generate_literature_review(retrieved_papers, topic)
            
            # Generate research recommendations
            recommendations = self.generator.generate_research_recommendations(retrieved_papers, topic)
            
            return {
                "topic": topic,
                "literature_review": literature_review,
                "research_recommendations": recommendations,
                "papers_analyzed": len(retrieved_papers),
                "papers": retrieved_papers
            }
            
        except Exception as e:
            logger.error(f"Error generating literature review: {e}")
            return {"error": f"Literature review generation failed: {str(e)}"}

    def identify_research_gaps(self, topic: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        Identify research gaps for a specific topic.
        
        Args:
            topic: Research topic
            max_hops: Maximum hops for graph traversal
            
        Returns:
            Research gap analysis results
        """
        try:
            # Get research gaps from graph
            gaps_data = self.retriever.get_research_gaps(topic, max_hops)
            
            if "error" in gaps_data:
                return gaps_data
            
            # Generate detailed gap analysis
            gap_analysis = self.generator.generate_gap_analysis(gaps_data)
            
            # Get related papers for context
            related_papers = self.retriever.hybrid_search(topic, alpha=0.5, limit=10)
            
            return {
                "topic": topic,
                "gap_analysis": gap_analysis,
                "gap_data": gaps_data,
                "related_papers": related_papers
            }
            
        except Exception as e:
            logger.error(f"Error identifying research gaps: {e}")
            return {"error": f"Research gap analysis failed: {str(e)}"}

    def track_methodology_evolution(self, method_name: str) -> Dict[str, Any]:
        """
        Track the evolution of a specific methodology.
        
        Args:
            method_name: Name of the methodology to track
            
        Returns:
            Methodology evolution analysis
        """
        try:
            # Get evolution data from graph
            evolution_data = self.retriever.get_methodology_evolution(method_name)
            
            # Generate evolution report
            evolution_report = self.generator.generate_methodology_evolution_report(evolution_data)
            
            return {
                "method": method_name,
                "evolution_report": evolution_report,
                "evolution_data": evolution_data
            }
            
        except Exception as e:
            logger.error(f"Error tracking methodology evolution: {e}")
            return {"error": f"Methodology evolution tracking failed: {str(e)}"}

    def analyze_collaboration_network(self, author_name: str = None) -> Dict[str, Any]:
        """
        Analyze collaboration networks.
        
        Args:
            author_name: Specific author to analyze (optional)
            
        Returns:
            Collaboration network analysis
        """
        try:
            # Get collaboration data from graph
            collaboration_data = self.retriever.get_collaboration_network(author_name)
            
            # Generate collaboration analysis
            collaboration_analysis = self.generator.generate_collaboration_analysis(collaboration_data)
            
            return {
                "author": author_name,
                "collaboration_analysis": collaboration_analysis,
                "collaboration_data": collaboration_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing collaboration network: {e}")
            return {"error": f"Collaboration analysis failed: {str(e)}"}

    def multi_hop_reasoning(self, query: str, max_hops: int = 3) -> Dict[str, Any]:
        """
        Perform multi-hop reasoning across the knowledge graph.
        
        Args:
            query: Research query
            max_hops: Maximum number of hops for reasoning
            
        Returns:
            Multi-hop reasoning results
        """
        try:
            # Perform graph search with multiple hops
            graph_papers = self.retriever.graph_search(query, max_hops=max_hops, limit=20)
            
            if not graph_papers:
                return {"error": "No results found for multi-hop reasoning"}
            
            # Generate comprehensive analysis
            research_summary = self.generator.generate_research_summary(graph_papers, query)
            recommendations = self.generator.generate_research_recommendations(graph_papers, query)
            
            return {
                "query": query,
                "max_hops": max_hops,
                "research_summary": research_summary,
                "research_recommendations": recommendations,
                "papers_found": len(graph_papers),
                "papers": graph_papers
            }
            
        except Exception as e:
            logger.error(f"Error in multi-hop reasoning: {e}")
            return {"error": f"Multi-hop reasoning failed: {str(e)}"}

    def close(self):
        """Close database connections."""
        if self.retriever:
            self.retriever.close() 