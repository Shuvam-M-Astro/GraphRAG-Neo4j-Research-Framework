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