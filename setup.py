"""
Setup script for Graph RAG Scientific Research project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graphrag-neo4j-research-framework",
    version="1.0.0",
    author="GraphRAG Neo4j Research Framework Team",
    author_email="team@graphrag-framework.com",
    description="GraphRAG Neo4j Research Framework - Complete system for scientific research and literature analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/graphrag/neo4j-research-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "graph-rag-init=src.database.init_database:main",
            "graph-rag-ingest=src.data_ingestion.arxiv_ingestion:main",
        ],
    },
) 