"""
Simplified Validation Utilities for GraphRAG Neo4j Research Framework
Provides essential Pydantic models and validation functions.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)

# Pydantic Models for Data Validation
class Paper(BaseModel):
    """Paper data model with essential validation."""
    paper_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=10, max_length=10000)
    year: int = Field(..., ge=1900, le=datetime.now().year + 1)
    journal: Optional[str] = Field(None, max_length=200)
    citations: int = Field(default=0, ge=0)
    authors: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    
    @validator('paper_id')
    def validate_paper_id(cls, v):
        if not v.strip():
            raise ValueError("Paper ID cannot be empty")
        return v.strip()
    
    @validator('abstract')
    def validate_abstract(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Abstract must be at least 10 characters long")
        return v.strip()

class SearchQuery(BaseModel):
    """Search query model with validation."""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=100, ge=1, le=1000)
    max_hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

class DatabaseConfig(BaseModel):
    """Database configuration model."""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")

class ModelConfig(BaseModel):
    """Model configuration model."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    llm_model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)

# Validation Functions
def validate_environment_variables() -> Dict[str, str]:
    """Validate required environment variables."""
    required_vars = {
        "NEO4J_URI": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "NEO4J_USER": os.getenv("NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", "password")
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return required_vars

def validate_paper_data(data: Dict[str, Any]) -> Paper:
    """Validate paper data using Pydantic model."""
    return Paper(**data)

def validate_search_params(query: str, max_results: int = 100, max_hops: int = 2, limit: int = 10) -> SearchQuery:
    """Validate search parameters using Pydantic model."""
    return SearchQuery(query=query, max_results=max_results, max_hops=max_hops, limit=limit)

def validate_database_config(uri: str = None, user: str = None, password: str = None) -> DatabaseConfig:
    """Validate database configuration using Pydantic model."""
    config_data = {}
    if uri:
        config_data["uri"] = uri
    if user:
        config_data["user"] = user
    if password:
        config_data["password"] = password
    
    return DatabaseConfig(**config_data)

def validate_model_config(embedding_model: str = None, llm_model: str = None, 
                         temperature: float = None, max_tokens: int = None) -> ModelConfig:
    """Validate model configuration using Pydantic model."""
    config_data = {}
    if embedding_model:
        config_data["embedding_model"] = embedding_model
    if llm_model:
        config_data["llm_model"] = llm_model
    if temperature is not None:
        config_data["temperature"] = temperature
    if max_tokens:
        config_data["max_tokens"] = max_tokens
    
    return ModelConfig(**config_data)

# Custom exception for validation errors
class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass 