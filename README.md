# GraphRAG Neo4j Research Framework

This project implements a Graph RAG (Retrieval-Augmented Generation) system specifically designed for scientific research and literature analysis. It connects research papers, authors, methodologies, and findings to enable sophisticated multi-hop reasoning and knowledge discovery.

## Features

- **GraphRAG Technology**: Combines graph traversal with vector similarity for multi-hop reasoning
- **Neo4j Integration**: Robust graph database for research relationships and knowledge graphs
- **Research Framework**: Complete system for scientific literature analysis and discovery
- **Multi-hop Reasoning**: Traverse relationships between papers, authors, methodologies, and findings
- **Research Gap Analysis**: Identify unexplored areas in research networks
- **Collaboration Pattern Discovery**: Find research collaboration networks and trends
- **Methodology Tracking**: Track how methodologies evolve across different research areas
- **Citation Network Analysis**: Analyze citation patterns and influence networks
- **Interactive Dashboard**: Streamlit-based interface for exploring the knowledge graph

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Neo4j Graph   │    │   RAG Pipeline  │
│                 │    │     Database     │    │                 │
│ • ArXiv API     │───▶│                 │───▶│ • Vector Store  │
│ • PubMed API    │    │ • Papers        │    │ • LLM Interface │
│ • Scholar API   │    │ • Authors       │    │ • Query Engine  │
│ • Custom Data   │    │ • Methods       │    │                 │
└─────────────────┘    │ • Citations     │    └─────────────────┘
                       │ • Keywords      │
                       └─────────────────┘
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Neo4j Database
- Install Neo4j Desktop or Neo4j Community Edition
- Create a new database
- Update `.env` file with your Neo4j credentials

### 3. Environment Configuration
Create a `.env` file:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

### 4. Initialize the Database
```bash
python src/database/init_database.py
```

### 5. Run the Application
```bash
streamlit run src/app/main.py
```

## Project Structure

```
├── src/
│   ├── database/           # Neo4j database operations
│   ├── data_ingestion/     # Data collection and processing
│   ├── graph_rag/          # Graph RAG core components
│   ├── visualization/      # Graph visualization tools
│   └── app/               # Streamlit web application
├── data/                  # Sample data and processed datasets
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Use Cases

### 1. Research Gap Analysis
- Find unexplored connections between research areas
- Identify missing methodologies in specific domains
- Discover potential collaboration opportunities

### 2. Literature Review Automation
- Automatically find relevant papers across multiple hops
- Generate comprehensive literature reviews
- Track methodology evolution over time

### 3. Collaboration Network Analysis
- Identify key researchers in specific fields
- Find potential collaborators based on research interests
- Analyze research community structures

### 4. Methodology Tracking
- Track how research methods spread across disciplines
- Identify successful methodology adaptations
- Find methodological gaps in research areas

## API Endpoints

The system provides REST API endpoints for:
- Paper search and retrieval
- Author analysis
- Methodology tracking
- Citation network analysis
- Research gap identification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details 