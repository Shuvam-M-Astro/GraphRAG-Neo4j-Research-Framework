@echo off
echo ========================================
echo GraphRAG Neo4j Research Framework Setup
echo ========================================
echo.

echo Step 1: Creating conda environment...
conda create -n graphrag python=3.11 -y

echo.
echo Step 2: Activating environment...
call conda activate graphrag

echo.
echo Step 3: Installing dependencies...
pip install -r requirements.txt

echo.
echo Step 4: Setting up environment file...
if not exist .env (
    copy env_example.txt .env
    echo Environment file created. Please edit .env with your API keys and database credentials.
) else (
    echo Environment file already exists.
)

echo.
echo Step 5: Verifying installation...
python -c "import neo4j, langchain, openai, pandas, numpy, torch; print('All dependencies installed successfully!')"

echo.
echo Step 6: Setting up Neo4j Database...
echo.
echo To start Neo4j database, run this command in a new terminal:
echo docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password123 neo4j:5.15.0
echo.
echo Or download Neo4j Desktop from: https://neo4j.com/download/
echo.

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Start Neo4j database (see instructions above)
echo 2. Edit .env file with your API keys and database credentials
echo 3. Activate the environment: conda activate graphrag
echo 4. Initialize database: python src/database/init_database.py
echo 5. Run the app: streamlit run src/app/main.py
echo 6. Or explore notebooks: jupyter notebook
echo.
echo To activate the environment in the future, run:
echo conda activate graphrag
echo.
pause 