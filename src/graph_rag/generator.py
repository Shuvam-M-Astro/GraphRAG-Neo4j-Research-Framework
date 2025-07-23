"""
Graph RAG Generator for Scientific Research
Generates comprehensive research insights using retrieved graph context.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json

load_dotenv()

logger = logging.getLogger(__name__)

class GraphRAGGenerator:
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the Graph RAG generator.
        
        Args:
            model_name: OpenAI model to use for generation
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            max_tokens=2000
        )
        
        # Define system prompts for different types of analysis
        self.system_prompts = {
            "research_summary": """You are an expert research analyst specializing in scientific literature analysis. 
            Your task is to provide comprehensive summaries of research findings based on graph-retrieved context.
            Focus on:
            1. Key findings and contributions
            2. Methodology connections and evolution
            3. Research gaps and opportunities
            4. Collaboration patterns
            5. Future research directions
            
            Always provide evidence-based insights and cite specific papers when possible.""",
            
            "gap_analysis": """You are a research gap analysis expert. Analyze the provided research context to identify:
            1. Underexplored research areas
            2. Missing methodology applications
            3. Potential collaboration opportunities
            4. Emerging research trends
            5. Cross-disciplinary opportunities
            
            Provide specific, actionable insights for researchers.""",
            
            "methodology_tracking": """You are a methodology evolution expert. Analyze how research methods:
            1. Spread across different disciplines
            2. Evolve over time
            3. Get adapted for new applications
            4. Influence other methodologies
            5. Create new research opportunities
            
            Focus on the trajectory and impact of specific methodologies.""",
            
            "collaboration_analysis": """You are a research collaboration network analyst. Examine:
            1. Key collaboration patterns
            2. Influential researchers and institutions
            3. Cross-institutional partnerships
            4. Emerging research communities
            5. Collaboration opportunities
            
            Identify strategic collaboration opportunities and research community dynamics."""
        }

    def generate_research_summary(self, retrieved_papers: List[Dict], query: str) -> str:
        """
        Generate a comprehensive research summary based on retrieved papers.
        
        Args:
            retrieved_papers: List of papers retrieved from the graph
            query: Original research query
            
        Returns:
            Generated research summary
        """
        # Prepare context from retrieved papers
        context = self._prepare_paper_context(retrieved_papers)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts["research_summary"]),
            ("human", f"""
            Research Query: {query}
            
            Retrieved Papers Context:
            {context}
            
            Please provide a comprehensive research summary that addresses the query and synthesizes insights from the retrieved papers.
            Include:
            1. Key findings and contributions
            2. Methodology connections
            3. Research trends and patterns
            4. Gaps and opportunities
            5. Future research directions
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating research summary: {e}")
            return f"Error generating summary: {str(e)}"

    def generate_gap_analysis(self, research_gaps: Dict[str, Any]) -> str:
        """
        Generate detailed gap analysis based on graph analysis.
        
        Args:
            research_gaps: Gap analysis data from graph retriever
            
        Returns:
            Generated gap analysis report
        """
        context = json.dumps(research_gaps, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts["gap_analysis"]),
            ("human", f"""
            Research Gap Analysis Data:
            {context}
            
            Please provide a detailed gap analysis report that includes:
            1. Identified research gaps with specific examples
            2. Methodology gaps and opportunities
            3. Collaboration opportunities
            4. Emerging research areas
            5. Strategic recommendations for researchers
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating gap analysis: {e}")
            return f"Error generating gap analysis: {str(e)}"

    def generate_methodology_evolution_report(self, evolution_data: Dict[str, Any]) -> str:
        """
        Generate methodology evolution analysis.
        
        Args:
            evolution_data: Methodology evolution data from graph retriever
            
        Returns:
            Generated evolution report
        """
        context = json.dumps(evolution_data, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts["methodology_tracking"]),
            ("human", f"""
            Methodology Evolution Data:
            {context}
            
            Please provide a detailed methodology evolution report that includes:
            1. Timeline of methodology development
            2. Cross-disciplinary applications
            3. Key adaptations and innovations
            4. Impact on research fields
            5. Future evolution predictions
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating methodology evolution report: {e}")
            return f"Error generating evolution report: {str(e)}"

    def generate_collaboration_analysis(self, collaboration_data: Dict[str, Any]) -> str:
        """
        Generate collaboration network analysis.
        
        Args:
            collaboration_data: Collaboration network data from graph retriever
            
        Returns:
            Generated collaboration analysis
        """
        context = json.dumps(collaboration_data, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompts["collaboration_analysis"]),
            ("human", f"""
            Collaboration Network Data:
            {context}
            
            Please provide a detailed collaboration analysis that includes:
            1. Key collaboration patterns and trends
            2. Influential researchers and institutions
            3. Cross-institutional partnerships
            4. Emerging research communities
            5. Strategic collaboration recommendations
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating collaboration analysis: {e}")
            return f"Error generating collaboration analysis: {str(e)}"

    def generate_literature_review(self, retrieved_papers: List[Dict], topic: str) -> str:
        """
        Generate a comprehensive literature review.
        
        Args:
            retrieved_papers: List of papers retrieved from the graph
            topic: Research topic for the literature review
            
        Returns:
            Generated literature review
        """
        context = self._prepare_paper_context(retrieved_papers)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert academic writer specializing in literature reviews. 
            Create comprehensive, well-structured literature reviews that:
            1. Synthesize findings from multiple sources
            2. Identify research trends and patterns
            3. Highlight methodological approaches
            4. Identify gaps and future directions
            5. Provide critical analysis of the field
            
            Structure the review with clear sections and logical flow."""),
            ("human", f"""
            Topic: {topic}
            
            Retrieved Papers:
            {context}
            
            Please generate a comprehensive literature review that:
            1. Introduces the topic and its significance
            2. Synthesizes key findings from the papers
            3. Analyzes methodological approaches
            4. Identifies research trends and patterns
            5. Highlights gaps and future research directions
            6. Provides critical analysis and conclusions
            
            Structure the review with clear sections and subsections.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating literature review: {e}")
            return f"Error generating literature review: {str(e)}"

    def generate_research_recommendations(self, retrieved_papers: List[Dict], query: str) -> str:
        """
        Generate research recommendations based on retrieved context.
        
        Args:
            retrieved_papers: List of papers retrieved from the graph
            query: Original research query
            
        Returns:
            Generated research recommendations
        """
        context = self._prepare_paper_context(retrieved_papers)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research strategy expert. Provide actionable research recommendations that:
            1. Build on existing research gaps
            2. Suggest novel methodology combinations
            3. Identify collaboration opportunities
            4. Recommend future research directions
            5. Provide strategic guidance for researchers
            
            Focus on practical, implementable recommendations."""),
            ("human", f"""
            Research Query: {query}
            
            Current Research Context:
            {context}
            
            Please provide detailed research recommendations that include:
            1. Specific research questions to pursue
            2. Methodology recommendations
            3. Collaboration opportunities
            4. Resource and funding considerations
            5. Timeline and priority recommendations
            6. Risk assessment and mitigation strategies
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating research recommendations: {e}")
            return f"Error generating recommendations: {str(e)}"

    def generate_graph_aware_insights(self, retrieved_papers: List[Dict], graph_context: Dict[str, Any], 
                                    query: str) -> Dict[str, Any]:
        """
        Generate insights that leverage graph structure and relationships.
        
        Args:
            retrieved_papers: List of papers retrieved from the graph
            graph_context: Additional graph context (paths, relationships, etc.)
            query: Original research query
            
        Returns:
            Dictionary containing graph-aware insights
        """
        context = self._prepare_paper_context(retrieved_papers)
        graph_context_str = self._prepare_graph_context(graph_context)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in graph-based research analysis. Analyze the provided research context 
            considering the graph structure and relationships to provide deep insights.
            
            Focus on:
            1. Graph-based patterns and connections
            2. Influence propagation and knowledge flow
            3. Structural relationships between entities
            4. Network effects and clustering
            5. Graph-theoretic insights about research evolution"""),
            ("human", f"""
            Research Query: {query}
            
            Retrieved Papers Context:
            {context}
            
            Graph Structure Context:
            {graph_context_str}
            
            Please provide comprehensive graph-aware insights that leverage the structural relationships 
            and network properties of the research landscape.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return {
                "graph_insights": response.content,
                "graph_context": graph_context,
                "query": query
            }
        except Exception as e:
            logger.error(f"Error generating graph-aware insights: {e}")
            return {"error": f"Graph-aware insights generation failed: {str(e)}"}

    def generate_path_based_analysis(self, path_analysis: Dict[str, Any], query: str) -> str:
        """
        Generate analysis based on path analysis results.
        
        Args:
            path_analysis: Results from path analysis
            query: Original research query
            
        Returns:
            Generated path-based analysis
        """
        path_context = self._prepare_path_context(path_analysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in analyzing research influence and knowledge flow through path analysis.
            Analyze the provided path data to understand how knowledge and influence propagate through the research network.
            
            Focus on:
            1. Path strength and significance
            2. Influence propagation patterns
            3. Knowledge flow dynamics
            4. Network centrality and bottlenecks
            5. Strategic research positioning"""),
            ("human", f"""
            Research Query: {query}
            
            Path Analysis Context:
            {path_context}
            
            Please provide a comprehensive analysis of the path-based insights, explaining how knowledge 
            and influence flow through the research network and what this reveals about the research landscape.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating path-based analysis: {e}")
            return f"Error generating path analysis: {str(e)}"

    def generate_temporal_evolution_report(self, temporal_data: Dict[str, Any], topic: str) -> str:
        """
        Generate comprehensive temporal evolution report.
        
        Args:
            temporal_data: Temporal analysis results
            topic: Research topic
            
        Returns:
            Generated temporal evolution report
        """
        temporal_context = self._prepare_temporal_context(temporal_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in analyzing research evolution over time. Analyze the provided 
            temporal data to understand how research topics, methods, and collaborations evolve.
            
            Focus on:
            1. Temporal trends and patterns
            2. Research lifecycle stages
            3. Emergence and decline of topics
            4. Seasonal and cyclical patterns
            5. Future trajectory predictions"""),
            ("human", f"""
            Research Topic: {topic}
            
            Temporal Evolution Data:
            {temporal_context}
            
            Please provide a comprehensive temporal evolution report that explains how the research landscape 
            has evolved over time and what trends we can expect in the future.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating temporal evolution report: {e}")
            return f"Error generating temporal report: {str(e)}"

    def generate_network_analysis_report(self, network_data: Dict[str, Any], analysis_type: str) -> str:
        """
        Generate network analysis report for collaboration and influence networks.
        
        Args:
            network_data: Network analysis results
            analysis_type: Type of network analysis (collaboration, influence, citation)
            
        Returns:
            Generated network analysis report
        """
        network_context = self._prepare_network_context(network_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert in {analysis_type} network analysis. Analyze the provided 
            network data to understand the structure and dynamics of research networks.
            
            Focus on:
            1. Network topology and structure
            2. Centrality and influence measures
            3. Community detection and clustering
            4. Network resilience and robustness
            5. Strategic network positioning"""),
            ("human", f"""
            Network Analysis Type: {analysis_type}
            
            Network Data:
            {network_context}
            
            Please provide a comprehensive network analysis report that explains the structure, 
            dynamics, and strategic implications of the research network.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating network analysis report: {e}")
            return f"Error generating network report: {str(e)}"

    def generate_cross_domain_insights(self, domain_data: Dict[str, Any], query: str) -> str:
        """
        Generate insights about cross-domain research connections.
        
        Args:
            domain_data: Cross-domain analysis results
            query: Research query
            
        Returns:
            Generated cross-domain insights
        """
        domain_context = self._prepare_domain_context(domain_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in cross-domain research analysis. Analyze the provided 
            data to understand how research concepts and methods bridge different domains.
            
            Focus on:
            1. Cross-domain knowledge transfer
            2. Interdisciplinary collaboration patterns
            3. Methodological adaptation across domains
            4. Emerging interdisciplinary fields
            5. Innovation opportunities at domain boundaries"""),
            ("human", f"""
            Research Query: {query}
            
            Cross-Domain Data:
            {domain_context}
            
            Please provide comprehensive cross-domain insights that explain how research concepts 
            and methods connect across different domains and what opportunities this creates.
            """)
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            return response.content
        except Exception as e:
            logger.error(f"Error generating cross-domain insights: {e}")
            return f"Error generating cross-domain insights: {str(e)}"

    def _prepare_paper_context(self, papers: List[Dict]) -> str:
        """
        Prepare paper context for LLM input.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            context_part = f"Paper {i}:\n"
            context_part += f"Title: {paper.get('title', 'N/A')}\n"
            context_part += f"Year: {paper.get('year', 'N/A')}\n"
            context_part += f"Journal: {paper.get('journal', 'N/A')}\n"
            context_part += f"Citations: {paper.get('citations', 'N/A')}\n"
            
            if paper.get('abstract'):
                context_part += f"Abstract: {paper['abstract']}\n"
            
            if paper.get('authors'):
                context_part += f"Authors: {', '.join(paper['authors'])}\n"
            
            if paper.get('methods'):
                context_part += f"Methods: {', '.join(paper['methods'])}\n"
            
            if paper.get('keywords'):
                context_part += f"Keywords: {', '.join(paper['keywords'])}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts) 

    def _prepare_graph_context(self, graph_context: Dict[str, Any]) -> str:
        """Prepare graph context for LLM prompts."""
        if not graph_context:
            return "No additional graph context available."
        
        context_parts = []
        
        if "paths" in graph_context:
            context_parts.append(f"Number of paths analyzed: {len(graph_context['paths'])}")
            if graph_context.get("path_metrics"):
                metrics = graph_context["path_metrics"]
                context_parts.append(f"Path metrics: avg_length={metrics.get('avg_path_length', 0):.2f}, "
                                   f"avg_strength={metrics.get('avg_path_strength', 0):.2f}")
        
        if "influence_flow" in graph_context:
            flow = graph_context["influence_flow"]
            if "entity_distribution" in flow:
                context_parts.append(f"Entity distribution: {flow['entity_distribution']}")
            if "dominant_entity_type" in flow:
                context_parts.append(f"Dominant entity type: {flow['dominant_entity_type']}")
        
        if "propagation_results" in graph_context:
            prop = graph_context["propagation_results"]
            context_parts.append(f"Influence propagation: {len(prop.get('influenced_entities', []))} entities influenced")
        
        return "\n".join(context_parts) if context_parts else "Graph context available but no specific patterns identified."

    def _prepare_path_context(self, path_analysis: Dict[str, Any]) -> str:
        """Prepare path analysis context for LLM prompts."""
        if not path_analysis:
            return "No path analysis data available."
        
        context_parts = []
        
        if "paths" in path_analysis:
            context_parts.append(f"Total paths analyzed: {len(path_analysis['paths'])}")
            
            # Analyze path characteristics
            if path_analysis["paths"]:
                lengths = [p["length"] for p in path_analysis["paths"]]
                strengths = [p["strength"] for p in path_analysis["paths"]]
                context_parts.append(f"Path lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.2f}")
                context_parts.append(f"Path strengths: min={min(strengths):.3f}, max={max(strengths):.3f}, avg={sum(strengths)/len(strengths):.3f}")
        
        if "path_metrics" in path_analysis:
            metrics = path_analysis["path_metrics"]
            context_parts.append(f"Path metrics: {metrics}")
        
        if "influence_flow" in path_analysis:
            flow = path_analysis["influence_flow"]
            context_parts.append(f"Influence flow: {flow}")
        
        return "\n".join(context_parts)

    def _prepare_temporal_context(self, temporal_data: Dict[str, Any]) -> str:
        """Prepare temporal data context for LLM prompts."""
        if not temporal_data:
            return "No temporal data available."
        
        context_parts = []
        
        if "temporal_evolution" in temporal_data:
            evolution = temporal_data["temporal_evolution"]
            context_parts.append(f"Temporal evolution over {len(evolution)} time periods:")
            
            for year, metrics in evolution.items():
                context_parts.append(f"  {year}: total_flow={metrics.get('total_flow', 0):.2f}, "
                                   f"avg_flow={metrics.get('avg_flow', 0):.2f}, "
                                   f"flow_count={metrics.get('flow_count', 0)}")
        
        if "flow_patterns" in temporal_data:
            patterns = temporal_data["flow_patterns"]
            context_parts.append(f"Flow patterns: {len(patterns)} patterns identified")
        
        return "\n".join(context_parts)

    def _prepare_network_context(self, network_data: Dict[str, Any]) -> str:
        """Prepare network data context for LLM prompts."""
        if not network_data:
            return "No network data available."
        
        context_parts = []
        
        if "collaborations" in network_data:
            collabs = network_data["collaborations"]
            context_parts.append(f"Collaboration network: {len(collabs)} collaborations")
        
        if "centrality_measures" in network_data:
            centrality = network_data["centrality_measures"]
            context_parts.append(f"Centrality measures: {centrality}")
        
        if "communities" in network_data:
            communities = network_data["communities"]
            context_parts.append(f"Detected communities: {len(communities)} communities")
        
        return "\n".join(context_parts)

    def _prepare_domain_context(self, domain_data: Dict[str, Any]) -> str:
        """Prepare cross-domain data context for LLM prompts."""
        if not domain_data:
            return "No cross-domain data available."
        
        context_parts = []
        
        if "domains" in domain_data:
            domains = domain_data["domains"]
            context_parts.append(f"Cross-domain connections: {len(domains)} domains involved")
        
        if "bridging_concepts" in domain_data:
            bridges = domain_data["bridging_concepts"]
            context_parts.append(f"Bridging concepts: {len(bridges)} concepts identified")
        
        if "interdisciplinary_patterns" in domain_data:
            patterns = domain_data["interdisciplinary_patterns"]
            context_parts.append(f"Interdisciplinary patterns: {patterns}")
        
        return "\n".join(context_parts) 