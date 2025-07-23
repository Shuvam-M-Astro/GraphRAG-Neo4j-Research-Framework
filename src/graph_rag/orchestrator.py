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
import time
from typing import Tuple

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

    def comprehensive_graph_analysis(self, query: str, analysis_depth: str = "deep") -> Dict[str, Any]:
        """
        Perform comprehensive graph-based analysis combining multiple GraphRAG techniques.
        
        Args:
            query: Research query
            analysis_depth: Analysis depth ("basic", "standard", "deep")
            
        Returns:
            Comprehensive analysis results
        """
        try:
            results = {
                "query": query,
                "analysis_depth": analysis_depth,
                "timestamp": time.time(),
                "analysis_components": {}
            }
            
            # 1. Graph-aware retrieval
            logger.info("Performing graph-aware retrieval...")
            graph_aware_results = self.retriever.graph_aware_search(
                query, 
                entity_types=["Paper", "Author", "Method", "Keyword"],
                path_constraints={"min_year": 2015},
                limit=20
            )
            results["analysis_components"]["graph_aware_retrieval"] = graph_aware_results
            
            # 2. Temporal analysis
            logger.info("Performing temporal analysis...")
            temporal_results = self.retriever.temporal_graph_search(
                query, 
                time_window=(2010, 2024), 
                temporal_weight=0.3
            )
            results["analysis_components"]["temporal_analysis"] = temporal_results
            
            # 3. Path analysis (if deep analysis)
            if analysis_depth in ["standard", "deep"]:
                logger.info("Performing path analysis...")
                if graph_aware_results:
                    # Use top results for path analysis
                    top_papers = graph_aware_results[:3]
                    path_analysis = self.retriever.path_analysis_search(
                        source_entity=top_papers[0]["paper_id"],
                        target_entity=top_papers[-1]["paper_id"],
                        max_paths=5
                    )
                    results["analysis_components"]["path_analysis"] = path_analysis
            
            # 4. Knowledge flow analysis (if deep analysis)
            if analysis_depth == "deep":
                logger.info("Performing knowledge flow analysis...")
                flow_analysis = self.retriever.knowledge_flow_analysis(
                    query, 
                    time_period=(2015, 2024)
                )
                results["analysis_components"]["knowledge_flow"] = flow_analysis
            
            # 5. Generate comprehensive insights
            logger.info("Generating comprehensive insights...")
            all_papers = graph_aware_results + temporal_results
            graph_context = results["analysis_components"]
            
            comprehensive_insights = self.generator.generate_graph_aware_insights(
                all_papers, graph_context, query
            )
            results["comprehensive_insights"] = comprehensive_insights
            
            # 6. Generate specific analysis reports
            if analysis_depth in ["standard", "deep"]:
                if "path_analysis" in results["analysis_components"]:
                    path_report = self.generator.generate_path_based_analysis(
                        results["analysis_components"]["path_analysis"], query
                    )
                    results["path_analysis_report"] = path_report
                
                if "knowledge_flow" in results["analysis_components"]:
                    temporal_report = self.generator.generate_temporal_evolution_report(
                        results["analysis_components"]["knowledge_flow"], query
                    )
                    results["temporal_evolution_report"] = temporal_report
            
            logger.info(f"Comprehensive graph analysis completed for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive graph analysis failed: {e}")
            return {"error": f"Comprehensive analysis failed: {str(e)}"}

    def influence_propagation_analysis(self, seed_entities: List[str], 
                                     analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze influence propagation from seed entities through the knowledge graph.
        
        Args:
            seed_entities: List of seed entity identifiers
            analysis_type: Type of analysis ("basic", "comprehensive")
            
        Returns:
            Influence propagation analysis results
        """
        try:
            results = {
                "seed_entities": seed_entities,
                "analysis_type": analysis_type,
                "timestamp": time.time(),
                "propagation_analysis": {}
            }
            
            # 1. Influence propagation search
            logger.info("Performing influence propagation analysis...")
            propagation_results = self.retriever.influence_propagation_search(
                seed_entities, 
                propagation_steps=3, 
                influence_threshold=0.5
            )
            results["propagation_analysis"]["propagation_results"] = propagation_results
            
            # 2. Entity-centric analysis for each seed
            entity_analyses = {}
            for entity in seed_entities:
                logger.info(f"Performing entity-centric analysis for: {entity}")
                entity_analysis = self.retriever.entity_centric_search(
                    entity, 
                    entity_type="Paper",  # Default to Paper, could be enhanced to detect type
                    max_hops=2
                )
                entity_analyses[entity] = entity_analysis
            
            results["propagation_analysis"]["entity_analyses"] = entity_analyses
            
            # 3. Network analysis (if comprehensive)
            if analysis_type == "comprehensive":
                logger.info("Performing network analysis...")
                network_data = self.retriever.get_collaboration_network()
                network_report = self.generator.generate_network_analysis_report(
                    network_data, "influence"
                )
                results["propagation_analysis"]["network_analysis"] = {
                    "network_data": network_data,
                    "network_report": network_report
                }
            
            # 4. Generate influence insights
            logger.info("Generating influence insights...")
            influence_context = {
                "propagation_results": propagation_results,
                "entity_analyses": entity_analyses
            }
            
            influence_insights = self.generator.generate_graph_aware_insights(
                [], influence_context, f"Influence propagation from {', '.join(seed_entities)}"
            )
            results["influence_insights"] = influence_insights
            
            logger.info(f"Influence propagation analysis completed for {len(seed_entities)} seed entities")
            return results
            
        except Exception as e:
            logger.error(f"Influence propagation analysis failed: {e}")
            return {"error": f"Influence propagation analysis failed: {str(e)}"}

    def cross_domain_research_analysis(self, primary_domain: str, 
                                     secondary_domains: List[str]) -> Dict[str, Any]:
        """
        Analyze cross-domain research connections and interdisciplinary patterns.
        
        Args:
            primary_domain: Primary research domain
            secondary_domains: List of secondary domains to analyze
            
        Returns:
            Cross-domain analysis results
        """
        try:
            results = {
                "primary_domain": primary_domain,
                "secondary_domains": secondary_domains,
                "timestamp": time.time(),
                "cross_domain_analysis": {}
            }
            
            # 1. Domain-specific retrievals
            domain_results = {}
            for domain in [primary_domain] + secondary_domains:
                logger.info(f"Retrieving papers for domain: {domain}")
                domain_papers = self.retriever.graph_aware_search(
                    domain, 
                    entity_types=["Paper", "Method", "Keyword"],
                    limit=15
                )
                domain_results[domain] = domain_papers
            
            results["cross_domain_analysis"]["domain_results"] = domain_results
            
            # 2. Cross-domain path analysis
            logger.info("Performing cross-domain path analysis...")
            cross_domain_paths = {}
            
            for sec_domain in secondary_domains:
                if domain_results[primary_domain] and domain_results[sec_domain]:
                    # Find paths between domains
                    primary_papers = domain_results[primary_domain][:3]
                    secondary_papers = domain_results[sec_domain][:3]
                    
                    for p_paper in primary_papers:
                        for s_paper in secondary_papers:
                            path_analysis = self.retriever.path_analysis_search(
                                source_entity=p_paper["paper_id"],
                                target_entity=s_paper["paper_id"],
                                max_paths=3
                            )
                            key = f"{primary_domain}_to_{sec_domain}"
                            if key not in cross_domain_paths:
                                cross_domain_paths[key] = []
                            cross_domain_paths[key].append(path_analysis)
            
            results["cross_domain_analysis"]["cross_domain_paths"] = cross_domain_paths
            
            # 3. Bridging concept analysis
            logger.info("Analyzing bridging concepts...")
            bridging_concepts = self._identify_bridging_concepts(domain_results)
            results["cross_domain_analysis"]["bridging_concepts"] = bridging_concepts
            
            # 4. Generate cross-domain insights
            logger.info("Generating cross-domain insights...")
            cross_domain_data = {
                "domain_results": domain_results,
                "cross_domain_paths": cross_domain_paths,
                "bridging_concepts": bridging_concepts
            }
            
            cross_domain_insights = self.generator.generate_cross_domain_insights(
                cross_domain_data, 
                f"Cross-domain analysis between {primary_domain} and {', '.join(secondary_domains)}"
            )
            results["cross_domain_insights"] = cross_domain_insights
            
            logger.info(f"Cross-domain analysis completed for {primary_domain} and {len(secondary_domains)} secondary domains")
            return results
            
        except Exception as e:
            logger.error(f"Cross-domain research analysis failed: {e}")
            return {"error": f"Cross-domain analysis failed: {str(e)}"}

    def _identify_bridging_concepts(self, domain_results: Dict[str, List]) -> Dict[str, Any]:
        """Identify concepts that bridge different domains."""
        bridging_concepts = {
            "shared_methods": [],
            "shared_keywords": [],
            "shared_authors": [],
            "concept_overlap": {}
        }
        
        # Extract methods, keywords, and authors from each domain
        domain_concepts = {}
        for domain, papers in domain_results.items():
            methods = set()
            keywords = set()
            authors = set()
            
            for paper in papers:
                methods.update(paper.get("methods", []))
                keywords.update(paper.get("keywords", []))
                authors.update(paper.get("authors", []))
            
            domain_concepts[domain] = {
                "methods": methods,
                "keywords": keywords,
                "authors": authors
            }
        
        # Find overlaps between domains
        domains = list(domain_concepts.keys())
        for i, domain1 in enumerate(domains):
            for domain2 in domains[i+1:]:
                shared_methods = domain_concepts[domain1]["methods"] & domain_concepts[domain2]["methods"]
                shared_keywords = domain_concepts[domain1]["keywords"] & domain_concepts[domain2]["keywords"]
                shared_authors = domain_concepts[domain1]["authors"] & domain_concepts[domain2]["authors"]
                
                key = f"{domain1}_x_{domain2}"
                bridging_concepts["concept_overlap"][key] = {
                    "shared_methods": list(shared_methods),
                    "shared_keywords": list(shared_keywords),
                    "shared_authors": list(shared_authors)
                }
                
                bridging_concepts["shared_methods"].extend(list(shared_methods))
                bridging_concepts["shared_keywords"].extend(list(shared_keywords))
                bridging_concepts["shared_authors"].extend(list(shared_authors))
        
        # Remove duplicates
        bridging_concepts["shared_methods"] = list(set(bridging_concepts["shared_methods"]))
        bridging_concepts["shared_keywords"] = list(set(bridging_concepts["shared_keywords"]))
        bridging_concepts["shared_authors"] = list(set(bridging_concepts["shared_authors"]))
        
        return bridging_concepts

    def research_trend_analysis(self, topic: str, time_period: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Analyze research trends and evolution for a specific topic.
        
        Args:
            topic: Research topic to analyze
            time_period: Time period to analyze (start_year, end_year)
            
        Returns:
            Research trend analysis results
        """
        try:
            results = {
                "topic": topic,
                "time_period": time_period,
                "timestamp": time.time(),
                "trend_analysis": {}
            }
            
            # 1. Temporal graph search
            logger.info("Performing temporal trend analysis...")
            if not time_period:
                time_period = (2010, 2024)
            
            temporal_results = self.retriever.temporal_graph_search(
                topic, 
                time_window=time_period, 
                temporal_weight=0.4
            )
            results["trend_analysis"]["temporal_results"] = temporal_results
            
            # 2. Knowledge flow analysis
            logger.info("Analyzing knowledge flow...")
            flow_analysis = self.retriever.knowledge_flow_analysis(topic, time_period)
            results["trend_analysis"]["knowledge_flow"] = flow_analysis
            
            # 3. Methodology evolution tracking
            logger.info("Tracking methodology evolution...")
            methodology_evolution = self.retriever.get_methodology_evolution(topic)
            results["trend_analysis"]["methodology_evolution"] = methodology_evolution
            
            # 4. Generate trend insights
            logger.info("Generating trend insights...")
            trend_context = {
                "temporal_results": temporal_results,
                "knowledge_flow": flow_analysis,
                "methodology_evolution": methodology_evolution
            }
            
            trend_insights = self.generator.generate_temporal_evolution_report(
                trend_context, topic
            )
            results["trend_insights"] = trend_insights
            
            # 5. Generate research recommendations
            logger.info("Generating research recommendations...")
            recommendations = self.generator.generate_research_recommendations(
                temporal_results, f"Research trends for {topic}"
            )
            results["research_recommendations"] = recommendations
            
            logger.info(f"Research trend analysis completed for topic: {topic}")
            return results
            
        except Exception as e:
            logger.error(f"Research trend analysis failed: {e}")
            return {"error": f"Trend analysis failed: {str(e)}"}

    def close(self):
        """Close database connections."""
        if self.retriever:
            self.retriever.close() 