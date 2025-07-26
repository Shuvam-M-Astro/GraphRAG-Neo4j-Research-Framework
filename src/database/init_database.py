"""
Database initialization script for Graph RAG Scientific Research project.
Sets up Neo4j schema and constraints for research papers, authors, methods, etc.
"""

import os
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jDatabase:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

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

    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
                "CREATE CONSTRAINT author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
                "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.method_id IS UNIQUE",
                "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.keyword_id IS UNIQUE",
                "CREATE CONSTRAINT journal_id IF NOT EXISTS FOR (j:Journal) REQUIRE j.journal_id IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")

            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX paper_year IF NOT EXISTS FOR (p:Paper) ON (p.year)",
                "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE INDEX method_name IF NOT EXISTS FOR (m:Method) ON (m.name)",
                "CREATE INDEX keyword_text IF NOT EXISTS FOR (k:Keyword) ON (k.text)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")

    def create_sample_data(self):
        """Create sample research data for testing."""
        with self.driver.session() as session:
            # Sample papers with diverse topics
            papers = [
                # Machine Learning Papers
                {
                    "paper_id": "p1",
                    "title": "Deep Learning for Natural Language Processing",
                    "abstract": "A comprehensive survey of deep learning approaches in NLP, covering transformer architectures and attention mechanisms for text processing and language understanding.",
                    "year": 2023,
                    "journal": "Nature Machine Intelligence",
                    "citations": 150
                },
                {
                    "paper_id": "p2", 
                    "title": "Graph Neural Networks for Scientific Discovery",
                    "abstract": "Novel applications of GNNs in scientific research, demonstrating how graph-based learning can accelerate drug discovery and molecular property prediction.",
                    "year": 2023,
                    "journal": "Science",
                    "citations": 89
                },
                {
                    "paper_id": "p3",
                    "title": "Transformer Architecture in Computer Vision",
                    "abstract": "Adapting transformer models for visual tasks, including image classification, object detection, and video understanding with attention-based architectures.",
                    "year": 2022,
                    "journal": "CVPR",
                    "citations": 234
                },
                
                # ADHD Research Papers
                {
                    "paper_id": "p4",
                    "title": "Neurobiological Mechanisms of ADHD: A Comprehensive Review",
                    "abstract": "This review examines the neurobiological underpinnings of Attention Deficit Hyperactivity Disorder, including dopaminergic and noradrenergic systems, brain structure abnormalities, and genetic factors contributing to the disorder.",
                    "year": 2023,
                    "journal": "Nature Reviews Neuroscience",
                    "citations": 67
                },
                {
                    "paper_id": "p5",
                    "title": "Cognitive Behavioral Therapy for Adult ADHD: A Meta-Analysis",
                    "abstract": "Meta-analysis of 15 randomized controlled trials examining the efficacy of cognitive behavioral therapy in treating adult ADHD symptoms, including attention, impulsivity, and executive function improvements.",
                    "year": 2022,
                    "journal": "Journal of Clinical Psychology",
                    "citations": 45
                },
                {
                    "paper_id": "p6",
                    "title": "Pharmacological Treatment of ADHD in Children and Adolescents",
                    "abstract": "Systematic review of stimulant and non-stimulant medications for ADHD treatment, including methylphenidate, amphetamines, and atomoxetine, with safety and efficacy comparisons.",
                    "year": 2023,
                    "journal": "Pediatrics",
                    "citations": 123
                },
                
                # Psychology Research Papers
                {
                    "paper_id": "p7",
                    "title": "Depression and Anxiety: Neural Circuit Mechanisms",
                    "abstract": "Investigation of neural circuit mechanisms underlying depression and anxiety disorders, focusing on prefrontal cortex, amygdala, and hippocampus interactions in emotional regulation.",
                    "year": 2022,
                    "journal": "Nature Neuroscience",
                    "citations": 89
                },
                {
                    "paper_id": "p8",
                    "title": "Social Media Use and Adolescent Mental Health",
                    "abstract": "Longitudinal study examining the relationship between social media usage patterns and mental health outcomes in adolescents, including depression, anxiety, and self-esteem measures.",
                    "year": 2023,
                    "journal": "JAMA Pediatrics",
                    "citations": 156
                },
                
                # Medical Research Papers
                {
                    "paper_id": "p9",
                    "title": "COVID-19 Long-term Effects on Cognitive Function",
                    "abstract": "Prospective cohort study investigating long-term cognitive effects of COVID-19 infection, including memory, attention, and executive function assessments in recovered patients.",
                    "year": 2023,
                    "journal": "The Lancet",
                    "citations": 234
                },
                {
                    "paper_id": "p10",
                    "title": "Precision Medicine in Cancer Treatment",
                    "abstract": "Review of precision medicine approaches in oncology, including genomic profiling, targeted therapies, and personalized treatment strategies for various cancer types.",
                    "year": 2022,
                    "journal": "Nature Medicine",
                    "citations": 178
                }
            ]

            # Sample authors with diverse backgrounds
            authors = [
                {"author_id": "a1", "name": "Dr. Sarah Johnson", "institution": "MIT"},
                {"author_id": "a2", "name": "Prof. Michael Chen", "institution": "Stanford"},
                {"author_id": "a3", "name": "Dr. Emily Rodriguez", "institution": "UC Berkeley"},
                {"author_id": "a4", "name": "Dr. James Wilson", "institution": "Harvard Medical School"},
                {"author_id": "a5", "name": "Prof. Lisa Thompson", "institution": "Yale University"},
                {"author_id": "a6", "name": "Dr. Robert Kim", "institution": "Johns Hopkins"},
                {"author_id": "a7", "name": "Prof. Maria Garcia", "institution": "UCLA"},
                {"author_id": "a8", "name": "Dr. David Brown", "institution": "Columbia University"}
            ]

            # Sample methods across different fields
            methods = [
                {"method_id": "m1", "name": "Transformer", "category": "Neural Architecture"},
                {"method_id": "m2", "name": "Graph Neural Network", "category": "Graph Learning"},
                {"method_id": "m3", "name": "Attention Mechanism", "category": "Neural Component"},
                {"method_id": "m4", "name": "Cognitive Behavioral Therapy", "category": "Psychological Treatment"},
                {"method_id": "m5", "name": "Meta-Analysis", "category": "Statistical Method"},
                {"method_id": "m6", "name": "Randomized Controlled Trial", "category": "Clinical Research"},
                {"method_id": "m7", "name": "fMRI", "category": "Neuroimaging"},
                {"method_id": "m8", "name": "Genomic Sequencing", "category": "Molecular Biology"},
                {"method_id": "m9", "name": "Longitudinal Study", "category": "Research Design"},
                {"method_id": "m10", "name": "Machine Learning", "category": "Computational Method"}
            ]

            # Sample keywords across diverse topics
            keywords = [
                # Machine Learning
                {"keyword_id": "k1", "text": "deep learning"},
                {"keyword_id": "k2", "text": "natural language processing"},
                {"keyword_id": "k3", "text": "graph neural networks"},
                {"keyword_id": "k4", "text": "computer vision"},
                {"keyword_id": "k5", "text": "transformer"},
                
                # ADHD and Psychology
                {"keyword_id": "k6", "text": "adhd"},
                {"keyword_id": "k7", "text": "attention deficit hyperactivity disorder"},
                {"keyword_id": "k8", "text": "cognitive behavioral therapy"},
                {"keyword_id": "k9", "text": "depression"},
                {"keyword_id": "k10", "text": "anxiety"},
                {"keyword_id": "k11", "text": "mental health"},
                {"keyword_id": "k12", "text": "neurobiology"},
                {"keyword_id": "k13", "text": "psychology"},
                
                # Medical Research
                {"keyword_id": "k14", "text": "covid-19"},
                {"keyword_id": "k15", "text": "cancer"},
                {"keyword_id": "k16", "text": "precision medicine"},
                {"keyword_id": "k17", "text": "pharmacology"},
                {"keyword_id": "k18", "text": "clinical trial"},
                {"keyword_id": "k19", "text": "neuroscience"},
                {"keyword_id": "k20", "text": "social media"}
            ]

            # Create nodes using MERGE to handle duplicates
            for paper in papers:
                session.run("""
                    MERGE (p:Paper {paper_id: $paper_id})
                    ON CREATE SET
                        p.title = $title,
                        p.abstract = $abstract,
                        p.year = $year,
                        p.journal = $journal,
                        p.citations = $citations
                """, paper)

            for author in authors:
                session.run("""
                    MERGE (a:Author {author_id: $author_id})
                    ON CREATE SET
                        a.name = $name,
                        a.institution = $institution
                """, author)

            for method in methods:
                session.run("""
                    MERGE (m:Method {method_id: $method_id})
                    ON CREATE SET
                        m.name = $name,
                        m.category = $category
                """, method)

            for keyword in keywords:
                session.run("""
                    MERGE (k:Keyword {keyword_id: $keyword_id})
                    ON CREATE SET
                        k.text = $text
                """, keyword)

            # Create relationships
            relationships = [
                # Paper-Author relationships
                ("p1", "a1", "AUTHORED_BY"),
                ("p1", "a2", "AUTHORED_BY"),
                ("p2", "a2", "AUTHORED_BY"),
                ("p2", "a3", "AUTHORED_BY"),
                ("p3", "a1", "AUTHORED_BY"),
                ("p3", "a3", "AUTHORED_BY"),
                ("p4", "a4", "AUTHORED_BY"),
                ("p4", "a5", "AUTHORED_BY"),
                ("p5", "a5", "AUTHORED_BY"),
                ("p5", "a6", "AUTHORED_BY"),
                ("p6", "a4", "AUTHORED_BY"),
                ("p6", "a7", "AUTHORED_BY"),
                ("p7", "a5", "AUTHORED_BY"),
                ("p7", "a8", "AUTHORED_BY"),
                ("p8", "a6", "AUTHORED_BY"),
                ("p8", "a7", "AUTHORED_BY"),
                ("p9", "a4", "AUTHORED_BY"),
                ("p9", "a8", "AUTHORED_BY"),
                ("p10", "a6", "AUTHORED_BY"),
                ("p10", "a7", "AUTHORED_BY"),
                
                # Paper-Method relationships
                ("p1", "m1", "USES_METHOD"),
                ("p1", "m3", "USES_METHOD"),
                ("p1", "m10", "USES_METHOD"),
                ("p2", "m2", "USES_METHOD"),
                ("p2", "m10", "USES_METHOD"),
                ("p3", "m1", "USES_METHOD"),
                ("p3", "m3", "USES_METHOD"),
                ("p4", "m7", "USES_METHOD"),
                ("p4", "m9", "USES_METHOD"),
                ("p5", "m4", "USES_METHOD"),
                ("p5", "m5", "USES_METHOD"),
                ("p6", "m6", "USES_METHOD"),
                ("p6", "m17", "USES_METHOD"),
                ("p7", "m7", "USES_METHOD"),
                ("p7", "m9", "USES_METHOD"),
                ("p8", "m9", "USES_METHOD"),
                ("p9", "m9", "USES_METHOD"),
                ("p10", "m8", "USES_METHOD"),
                ("p10", "m10", "USES_METHOD"),
                
                # Paper-Keyword relationships
                ("p1", "k1", "HAS_KEYWORD"),
                ("p1", "k2", "HAS_KEYWORD"),
                ("p1", "k5", "HAS_KEYWORD"),
                ("p2", "k1", "HAS_KEYWORD"),
                ("p2", "k3", "HAS_KEYWORD"),
                ("p3", "k1", "HAS_KEYWORD"),
                ("p3", "k4", "HAS_KEYWORD"),
                ("p3", "k5", "HAS_KEYWORD"),
                ("p4", "k6", "HAS_KEYWORD"),
                ("p4", "k7", "HAS_KEYWORD"),
                ("p4", "k12", "HAS_KEYWORD"),
                ("p4", "k19", "HAS_KEYWORD"),
                ("p5", "k6", "HAS_KEYWORD"),
                ("p5", "k7", "HAS_KEYWORD"),
                ("p5", "k8", "HAS_KEYWORD"),
                ("p5", "k13", "HAS_KEYWORD"),
                ("p6", "k6", "HAS_KEYWORD"),
                ("p6", "k7", "HAS_KEYWORD"),
                ("p6", "k17", "HAS_KEYWORD"),
                ("p6", "k18", "HAS_KEYWORD"),
                ("p7", "k9", "HAS_KEYWORD"),
                ("p7", "k10", "HAS_KEYWORD"),
                ("p7", "k12", "HAS_KEYWORD"),
                ("p7", "k19", "HAS_KEYWORD"),
                ("p8", "k11", "HAS_KEYWORD"),
                ("p8", "k13", "HAS_KEYWORD"),
                ("p8", "k20", "HAS_KEYWORD"),
                ("p9", "k14", "HAS_KEYWORD"),
                ("p9", "k19", "HAS_KEYWORD"),
                ("p10", "k15", "HAS_KEYWORD"),
                ("p10", "k16", "HAS_KEYWORD"),
                ("p10", "k18", "HAS_KEYWORD"),
                
                # Citation relationships
                ("p2", "p1", "CITES"),
                ("p3", "p1", "CITES"),
                ("p5", "p4", "CITES"),
                ("p6", "p4", "CITES"),
                ("p7", "p4", "CITES"),
                ("p8", "p7", "CITES"),
                ("p9", "p7", "CITES"),
                ("p10", "p9", "CITES"),
                
                # Author collaboration
                ("a1", "a2", "COLLABORATED_WITH"),
                ("a2", "a3", "COLLABORATED_WITH"),
                ("a1", "a3", "COLLABORATED_WITH"),
                ("a4", "a5", "COLLABORATED_WITH"),
                ("a5", "a6", "COLLABORATED_WITH"),
                ("a6", "a7", "COLLABORATED_WITH"),
                ("a7", "a8", "COLLABORATED_WITH")
            ]

            for source, target, relationship in relationships:
                session.run(f"""
                    MATCH (source), (target)
                    WHERE source.paper_id = $source_id OR source.author_id = $source_id
                    AND target.paper_id = $target_id OR target.author_id = $target_id
                    MERGE (source)-[r:{relationship}]->(target)
                """, {"source_id": source, "target_id": target})

            logger.info("Sample data created successfully")

def main():
    """Main function to initialize the database."""
    db = Neo4jDatabase()
    
    try:
        db.connect()
        db.create_constraints_and_indexes()
        db.create_sample_data()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main() 