# GraphRAG Neo4j Research Framework - Project Structure

## Overview
This project implements the GraphRAG Neo4j Research Framework - a comprehensive Graph RAG (Retrieval-Augmented Generation) system specifically designed for scientific research and literature analysis. It combines Neo4j graph database, vector embeddings, and LLM generation to provide sophisticated research insights.

## Directory Structure

```
GraphRAG Neo4j Research Framework/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup script
├── env_example.txt                    # Environment variables template
├── test_system.py                     # System test script
├── PROJECT_STRUCTURE.md               # This file
│
├── src/                               # Main source code
│   ├── __init__.py                    # Package initialization
│   │
│   ├── database/                      # Database operations
│   │   ├── __init__.py
│   │   └── init_database.py          # Neo4j schema and sample data
│   │
│   ├── data_ingestion/               # Data collection and processing
│   │   ├── __init__.py
│   │   └── arxiv_ingestion.py        # ArXiv paper ingestion
│   │
│   ├── graph_rag/                    # Core Graph RAG components
│   │   ├── __init__.py
│   │   ├── graph_retriever.py        # Graph-based retrieval
│   │   ├── generator.py              # LLM generation
│   │   └── orchestrator.py           # Main orchestrator
│   │
│   ├── visualization/                 # Graph visualization tools
│   │   ├── __init__.py
│   │   └── graph_visualizer.py       # Interactive visualizations
│   │
│   └── app/                          # Streamlit web application
│       ├── __init__.py
│       └── main.py                   # Main web interface
│
├── notebooks/                         # Jupyter notebooks
│   └── 01_basic_usage.ipynb          # Basic usage examples
│
├── data/                             # Data storage (created at runtime)
│   └── vector_db/                    # FAISS vector database
│
└── docs/                             # Documentation (to be created)
```

## Core Components

### 1. Database Layer (`src/database/`)
- **Neo4j Integration**: Graph database for storing research entities and relationships
- **Schema Management**: Constraints, indexes, and data models
- **Sample Data**: Initial research papers, authors, methods, and relationships

### 2. Data Ingestion (`src/data_ingestion/`)
- **ArXiv Integration**: Fetches papers from ArXiv API
- **Entity Extraction**: Keywords, methodologies, and author information
- **Relationship Creation**: Citations, collaborations, and method usage

### 3. Graph RAG Core (`src/graph_rag/`)
- **GraphRetriever**: Combines vector similarity with graph traversal
- **GraphRAGGenerator**: LLM-based content generation
- **GraphRAGOrchestrator**: Coordinates retrieval and generation

### 4. Visualization (`src/visualization/`)
- **Network Graphs**: Collaboration and citation networks
- **Trend Analysis**: Research trends over time
- **Interactive Plots**: Plotly-based visualizations

### 5. Web Application (`src/app/`)
- **Streamlit Interface**: User-friendly web application
- **Multiple Analysis Types**: Research topic, gap analysis, methodology tracking
- **Real-time Results**: Interactive visualizations and reports

## Key Features

### Research Analysis Capabilities
1. **Multi-hop Reasoning**: Traverse relationships across papers, authors, and methods
2. **Literature Review Generation**: Automated comprehensive literature reviews
3. **Research Gap Analysis**: Identify unexplored areas and opportunities
4. **Methodology Evolution Tracking**: Monitor how methods spread and evolve
5. **Collaboration Network Analysis**: Discover research communities and partnerships

### Technical Features
1. **Hybrid Search**: Combines vector similarity with graph structure
2. **Real-time Processing**: Dynamic retrieval and generation
3. **Scalable Architecture**: Modular design for easy extension
4. **Interactive Visualizations**: Rich, interactive graphs and charts
5. **Comprehensive API**: Programmatic access to all features

## Data Model

### Neo4j Graph Schema
```
(Paper)-[:AUTHORED_BY]->(Author)
(Paper)-[:HAS_KEYWORD]->(Keyword)
(Paper)-[:USES_METHOD]->(Method)
(Paper)-[:CITES]->(Paper)
(Author)-[:COLLABORATED_WITH]->(Author)
(Paper)-[:BELONGS_TO]->(Category)
```

### Entity Types
- **Papers**: Research publications with metadata
- **Authors**: Researchers and their affiliations
- **Methods**: Research methodologies and techniques
- **Keywords**: Research topics and concepts
- **Categories**: Research domains and fields

## Usage Examples

### 1. Research Topic Analysis
```python
from src.graph_rag.orchestrator import GraphRAGOrchestrator

orchestrator = GraphRAGOrchestrator()
results = orchestrator.analyze_research_topic("deep learning in NLP")
```

### 2. Literature Review Generation
```python
lit_review = orchestrator.generate_literature_review("transformer architecture", max_papers=20)
```

### 3. Research Gap Analysis
```python
gaps = orchestrator.identify_research_gaps("graph neural networks", max_hops=3)
```

### 4. Methodology Evolution
```python
evolution = orchestrator.track_methodology_evolution("transformer")
```

### 5. Multi-hop Reasoning
```python
reasoning = orchestrator.multi_hop_reasoning("How do transformers influence computer vision?", max_hops=3)
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd graph-rag-project

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env_example.txt .env
# Edit .env with your credentials
```

### 2. Database Setup
```bash
# Initialize Neo4j database
python src/database/init_database.py

# Ingest sample data
python src/data_ingestion/arxiv_ingestion.py
```

### 3. Run the Application
```bash
# Start the web application
streamlit run src/app/main.py

# Or run tests
python test_system.py
```

## Configuration

### Environment Variables
- `NEO4J_URI`: Neo4j database connection URI
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `OPENAI_API_KEY`: OpenAI API key for generation
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4)

### Database Configuration
- **Neo4j Version**: 5.x or later
- **APOC Procedures**: Required for advanced graph operations
- **Memory**: Recommended 4GB+ for production use

## Performance Considerations

### Scalability
- **Vector Index**: FAISS for efficient similarity search
- **Graph Indexes**: Neo4j indexes for fast traversal
- **Caching**: Streamlit caching for repeated queries

### Optimization
- **Batch Processing**: Efficient data ingestion
- **Query Optimization**: Optimized Cypher queries
- **Memory Management**: Proper connection handling

## Extensibility

### Adding New Data Sources
1. Create new ingestion module in `src/data_ingestion/`
2. Implement data extraction and storage methods
3. Update schema if needed

### Adding New Analysis Types
1. Extend `GraphRAGOrchestrator` with new methods
2. Create corresponding generator prompts
3. Add visualization components

### Custom Visualizations
1. Extend `GraphVisualizer` class
2. Create new Plotly figures
3. Integrate with Streamlit interface

## Troubleshooting

### Common Issues
1. **Neo4j Connection**: Check URI, credentials, and network
2. **OpenAI API**: Verify API key and model availability
3. **Memory Issues**: Increase Neo4j memory allocation
4. **Import Errors**: Check Python path and dependencies

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Install dev dependencies: `pip install -e .[dev]`
4. Run tests: `pytest`
5. Submit pull request

### Code Style
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create new issue with detailed description
4. Contact maintainers for critical issues 