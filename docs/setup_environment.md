# GraphRAG Neo4j Research Framework - Environment Setup Guide

This guide will help you set up an Anaconda environment for the GraphRAG Neo4j Research Framework project.

## Prerequisites

- Anaconda or Miniconda installed on your system
- Git (to clone the repository if not already done)
- Neo4j Database (optional, for full functionality)
- Docker (optional, for running Neo4j in a container)

## Step-by-Step Setup Instructions

### 1. Open Anaconda Prompt

- Press `Win + R`, type `cmd`, and press Enter
- Or search for "Anaconda Prompt" in the Start menu
- Navigate to your project directory:
  ```bash
  cd "path\to\directory"
  ```

### 2. Create a New Conda Environment

```bash
conda create -n graphrag python=3.11 -y
```

### 3. Activate the Environment

```bash
conda activate graphrag
```

### 4. Install Dependencies

#### Option A: Install from requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Install the package in development mode
```bash
pip install -e .
```

### 5. Set Up Environment Variables

1. Copy the environment example file:
   ```bash
   copy env_example.txt .env
   ```

2. Edit the `.env` file with your actual configuration:
   - Update Neo4j credentials if you have a local Neo4j instance
   - Add your OpenAI API key
   - Adjust other settings as needed

### 6. Set Up Neo4j Database

#### Option A: Using Docker (Recommended for quick setup)
```bash
# Run Neo4j in a Docker container
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5.15.0
```

**Note**: Keep this terminal window open. The Neo4j database will be accessible at:
- Web interface: http://localhost:7474
- Bolt connection: bolt://localhost:7687
- Username: neo4j
- Password: password123

#### Option B: Using Neo4j Desktop
1. Download Neo4j Desktop from https://neo4j.com/download/
2. Install and create a new database
3. Set your password and start the database

#### Option C: Using Neo4j Community Edition
1. Download Neo4j Community Edition from https://neo4j.com/download/
2. Install and configure the service
3. Start the Neo4j service

### 7. Verify Installation

```bash
python -c "import neo4j, langchain, openai, pandas, numpy, torch; print('All dependencies installed successfully!')"
```

### 8. Test Neo4j Connection

```bash
python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password123'))
try:
    with driver.session() as session:
        result = session.run('RETURN 1 as test')
        print('Neo4j connection successful:', result.single()['test'])
    driver.close()
except Exception as e:
    print('Neo4j connection failed:', e)
"
```

## Alternative Setup Methods

### Using conda-forge for Core Dependencies

```bash
# Create environment with conda-forge
conda create -n graphrag python=3.11 -c conda-forge -y
conda activate graphrag

# Install core scientific packages via conda
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn networkx plotly requests beautifulsoup4 -y

# Install remaining packages via pip
pip install -r requirements.txt
```

### Using conda environment.yml (Alternative)

If you prefer using a conda environment file, you can create one:

```yaml
name: graphrag
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - networkx
  - plotly
  - requests
  - beautifulsoup4
  - pip
  - pip:
    - neo4j==5.15.0
    - langchain==0.1.0
    - langchain-community==0.0.10
    - langchain-openai==0.0.5
    - openai==1.3.7
    - python-dotenv==1.0.0
    - streamlit==1.29.0
    - arxiv==2.1.0
    - scholarly==1.7.11
    - py2neo==2021.2.4
    - transformers==4.36.2
    - torch==2.1.2
    - sentence-transformers==2.2.2
    - faiss-cpu==1.7.4
    - tiktoken==0.5.2
    - pydantic==2.5.2
    - mypy==1.8.0
    - typeguard==4.0.0
```

Then install with:
```bash
conda env create -f environment.yml
conda activate graphrag
```

## Running the Application

### 1. Start Neo4j Database (if not already running)
```bash
# Using Docker
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5.15.0
```

### 2. Initialize the Database (if using Neo4j)
```bash
python src/database/init_database.py
```

### 3. Run the Streamlit App
```bash
streamlit run src/app/main.py
```

### 4. Run Jupyter Notebooks
```bash
jupyter notebook
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: If you encounter CUDA issues with PyTorch, install the CPU-only version:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Version Conflicts**: If you encounter version conflicts, try:
   ```bash
   conda clean --all
   pip cache purge
   pip install -r requirements.txt --force-reinstall
   ```

3. **Neo4j Connection Issues**: Make sure Neo4j is running and accessible at the configured URI.

### Environment Management

- **List environments**: `conda env list`
- **Remove environment**: `conda env remove -n graphrag`
- **Export environment**: `conda env export > environment.yml`
- **Deactivate**: `conda deactivate`

## Next Steps

1. Configure your `.env` file with appropriate API keys and database credentials
2. Run the database initialization script
3. Start exploring the notebooks in the `notebooks/` directory
4. Run the Streamlit application for the web interface

## Support

If you encounter any issues during setup, please:
1. Check the troubleshooting section above
2. Ensure all prerequisites are installed
3. Verify your Python version (3.8+ required)
4. Check the project's README.md for additional information 