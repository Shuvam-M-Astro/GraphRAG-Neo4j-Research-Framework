# GraphRAG Neo4j Research Framework

This project implements a Graph RAG (Retrieval-Augmented Generation) system specifically designed for scientific research and literature analysis. It connects research papers, authors, methodologies, and findings to enable sophisticated multi-hop reasoning and knowledge discovery.

![GraphRAG Research Framework](images/1.png)

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

## Quick Start with Docker

### 1. Start Neo4j Database with Docker
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted=apoc.* \
  neo4j:latest
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
OPENAI_API_KEY=your_openai_api_key
```

### 4. Initialize the Database
```bash
python src/database/init_database.py
```

### 5. Test the System
```bash
python test_suite.py
```

### 6. Run the Web Application
```bash
streamlit run src/app/main.py
```

## Viewing the Graph Database

### Option 1: Neo4j Browser (Recommended)
1. Open your browser and go to: `http://localhost:7474`
2. Login with credentials: `neo4j` / `password123`
3. Run these Cypher queries to explore the graph:

```cypher
// View all nodes and relationships
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50

// Count total nodes
MATCH (n) RETURN count(n) as total_nodes

// View papers
MATCH (p:Paper) RETURN p.paper_id, p.title, p.year

// View authors
MATCH (a:Author) RETURN a.name, a.institution

// View relationships
MATCH ()-[r]->() RETURN type(r), count(r) as count

// Visualize the graph
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 20
```

### Option 2: Python Scripts
```bash
# View database contents
python view_database.py

# Visualize graph (requires networkx, matplotlib)
python visualize_graph.py
```

### Option 3: Streamlit Dashboard
The web application includes interactive visualizations:
- Graph visualization page
- Research trends
- Collaboration networks
- Citation networks

## Advanced Cypher Queries for Graph Analysis

### Research Network Analysis

```cypher
// Find the most influential papers by citation count
MATCH (p:Paper)
RETURN p.title, p.citations, p.year
ORDER BY p.citations DESC
LIMIT 10

// Find papers that bridge different research areas
MATCH (p:Paper)-[:HAS_KEYWORD]->(k1:Keyword)
MATCH (p)-[:HAS_KEYWORD]->(k2:Keyword)
WHERE k1.text <> k2.text
WITH p, collect(DISTINCT k1.text + k2.text) as keyword_pairs
RETURN p.title, keyword_pairs
ORDER BY size(keyword_pairs) DESC
LIMIT 10

// Find emerging research trends (papers from last 2 years with high citations)
MATCH (p:Paper)
WHERE p.year >= 2022 AND p.citations > 10
RETURN p.title, p.year, p.citations, p.journal
ORDER BY p.citations DESC
```

### Collaboration Network Analysis

```cypher
// Find the most collaborative authors
MATCH (a:Author)-[:COLLABORATED_WITH]-(b:Author)
WITH a.name as author, count(b) as collaborators
RETURN author, collaborators
ORDER BY collaborators DESC
LIMIT 15

// Find research communities (connected components)
MATCH (a:Author)-[:COLLABORATED_WITH]-(b:Author)
WITH a, b
CALL gds.alpha.scc.stream('Author', 'COLLABORATED_WITH')
YIELD nodeId, componentId
RETURN componentId, count(nodeId) as community_size
ORDER BY community_size DESC

// Find potential collaborators for an author
MATCH (author:Author {name: "John Smith"})-[:COLLABORATED_WITH]-(collab:Author)-[:COLLABORATED_WITH]-(potential:Author)
WHERE NOT (author)-[:COLLABORATED_WITH]-(potential)
RETURN potential.name, count(collab) as mutual_collaborators
ORDER BY mutual_collaborators DESC
LIMIT 10
```

### Methodology Evolution Tracking

```cypher
// Track how a methodology spreads across different fields
MATCH (p:Paper)-[:USES_METHOD]->(m:Method {name: "Machine Learning"})
MATCH (p)-[:HAS_KEYWORD]->(k:Keyword)
WITH m.name as method, k.text as field, p.year as year
RETURN method, field, year, count(*) as usage_count
ORDER BY year, field

// Find methodologies that are gaining popularity
MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
WHERE p.year >= 2020
WITH m.name as method, p.year as year, count(*) as usage
RETURN method, year, usage
ORDER BY method, year

// Find interdisciplinary methodology applications
MATCH (p1:Paper)-[:USES_METHOD]->(m:Method)
MATCH (p2:Paper)-[:USES_METHOD]->(m)
MATCH (p1)-[:HAS_KEYWORD]->(k1:Keyword)
MATCH (p2)-[:HAS_KEYWORD]->(k2:Keyword)
WHERE k1.text <> k2.text
RETURN m.name as method, k1.text as field1, k2.text as field2, count(*) as applications
ORDER BY applications DESC
```

### Citation Network Analysis

```cypher
// Find citation chains (papers that cite papers that cite...)
MATCH path = (start:Paper)-[:CITES*1..3]->(end:Paper)
WHERE start.paper_id = "paper_123"
RETURN length(path) as chain_length, end.title
ORDER BY chain_length

// Find papers that are highly cited but don't cite many others
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
WITH p, count(cited) as outgoing_citations
WHERE p.citations > 50 AND outgoing_citations < 10
RETURN p.title, p.citations, outgoing_citations
ORDER BY p.citations DESC

// Find citation reciprocity
MATCH (p1:Paper)-[:CITES]->(p2:Paper)
MATCH (p2)-[:CITES]->(p1)
RETURN p1.title as paper1, p2.title as paper2
```

### Research Gap Analysis

```cypher
// Find keywords that appear together but have few connecting papers
MATCH (k1:Keyword)<-[:HAS_KEYWORD]-(p:Paper)-[:HAS_KEYWORD]->(k2:Keyword)
WHERE k1.text < k2.text
WITH k1.text as keyword1, k2.text as keyword2, count(p) as connection_strength
WHERE connection_strength < 3
RETURN keyword1, keyword2, connection_strength
ORDER BY connection_strength

// Find authors who work in isolated research areas
MATCH (a:Author)-[:AUTHORED_BY]->(p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
WITH a.name as author, collect(DISTINCT k.text) as research_areas
WHERE size(research_areas) = 1
RETURN author, research_areas

// Find research areas with few methodologies
MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Paper)
OPTIONAL MATCH (p)-[:USES_METHOD]->(m:Method)
WITH k.text as research_area, count(DISTINCT m.name) as method_count
WHERE method_count < 3
RETURN research_area, method_count
ORDER BY method_count
```

### Advanced Graph Analytics

```cypher
// Find the most central papers in the citation network
MATCH (p:Paper)
OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
WITH p, count(cited) as outgoing, count(citing) as incoming
RETURN p.title, outgoing, incoming, (outgoing + incoming) as centrality
ORDER BY centrality DESC
LIMIT 10

// Find research clusters using keyword similarity
MATCH (p1:Paper)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(p2:Paper)
WHERE p1.paper_id < p2.paper_id
WITH p1, p2, count(k) as shared_keywords
WHERE shared_keywords >= 2
RETURN p1.title as paper1, p2.title as paper2, shared_keywords
ORDER BY shared_keywords DESC

// Find temporal research patterns
MATCH (p:Paper)-[:HAS_KEYWORD]->(k:Keyword)
WITH k.text as keyword, p.year as year, count(*) as paper_count
WHERE year >= 2015
RETURN keyword, year, paper_count
ORDER BY keyword, year
```

### Performance and Optimization Queries

```cypher
// Create indexes for better performance
CREATE INDEX paper_id_index FOR (p:Paper) ON (p.paper_id);
CREATE INDEX author_name_index FOR (a:Author) ON (a.name);
CREATE INDEX keyword_text_index FOR (k:Keyword) ON (k.text);
CREATE INDEX method_name_index FOR (m:Method) ON (m.name);

// Analyze database statistics
MATCH (n) RETURN labels(n) as node_type, count(n) as count
ORDER BY count DESC;

MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count
ORDER BY count DESC;

// Find orphaned nodes
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n) as node_type, count(n) as orphan_count;
```

## Manual Setup (Alternative to Docker)

### 1. Install Neo4j Desktop
- Download from [Neo4j Desktop](https://neo4j.com/download/)
- Create a new database
- Start the database service

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file with your Neo4j credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_openai_api_key
```

### 4. Initialize and Test
```bash
# Initialize database
python src/database/init_database.py

# Test the system
python test_system.py

# Run the application
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