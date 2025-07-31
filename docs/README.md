# Documentation

This directory contains comprehensive documentation for the GraphRAG Neo4j Research Framework.

## 📚 Documentation Index

### Setup and Installation
- **[Setup Environment](setup_environment.md)** - Complete setup guide for the development environment
- **[Project Structure](PROJECT_STRUCTURE.md)** - Detailed architecture and component overview

### Performance and Optimization
- **[Performance Improvements](PERFORMANCE_IMPROVEMENTS.md)** - Optimization strategies and best practices
- **[Validation Guide](VALIDATION_GUIDE.md)** - Testing and validation procedures

### Troubleshooting and Support
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and solutions

## 🚀 Quick Reference

### Environment Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"

# 3. Initialize system
python scripts/system_setup.py

# 4. Start application
streamlit run src/app/main.py
```

### Common Commands
```bash
# Scrape ArXiv papers
python scripts/arxiv_scraper.py --query "machine learning" --max_results 50

# Build vector index
python scripts/vector_index_builder.py

# View database
python scripts/database_viewer.py

# Visualize graph
python scripts/graph_visualizer.py

# Run tests
python -m pytest tests/
```

### Key Configuration Files
- `requirements.txt` - Python dependencies
- `environment.yml` - Conda environment
- `mypy.ini` - Type checking configuration
- `setup.py` - Package installation

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── setup_environment.md         # Environment setup guide
├── PROJECT_STRUCTURE.md         # Architecture overview
├── PERFORMANCE_IMPROVEMENTS.md  # Performance optimization
├── VALIDATION_GUIDE.md          # Testing and validation
└── TROUBLESHOOTING.md           # Common issues and solutions
```

## 🔧 Development Workflow

1. **Setup**: Follow the setup environment guide
2. **Development**: Use the project structure guide for architecture
3. **Testing**: Use the validation guide for testing procedures
4. **Optimization**: Reference performance improvements for optimization
5. **Troubleshooting**: Check troubleshooting guide for common issues

## 📞 Getting Help

- **Setup Issues**: Check `setup_environment.md`
- **Performance Issues**: Review `PERFORMANCE_IMPROVEMENTS.md`
- **Testing Issues**: See `VALIDATION_GUIDE.md`
- **General Issues**: Consult `TROUBLESHOOTING.md`
- **GitHub Issues**: Open an issue on the repository 