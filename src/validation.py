"""
Validation utilities for GraphRAG Neo4j Research Framework
Provides Pydantic models and validation functions for data integrity.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import HttpUrl
import logging

logger = logging.getLogger(__name__)

# Custom validators
def validate_year(value: int) -> int:
    """Validate year is reasonable for research papers."""
    if not isinstance(value, int):
        raise ValueError("Year must be an integer")
    if value < 1900 or value > datetime.now().year + 1:
        raise ValueError(f"Year must be between 1900 and {datetime.now().year + 1}")
    return value

def validate_citations(value: int) -> int:
    """Validate citation count is non-negative."""
    if not isinstance(value, int):
        raise ValueError("Citations must be an integer")
    if value < 0:
        raise ValueError("Citations must be non-negative")
    return value

def validate_paper_id(value: str) -> str:
    """Validate paper ID format."""
    if not isinstance(value, str):
        raise ValueError("Paper ID must be a string")
    if not value.strip():
        raise ValueError("Paper ID cannot be empty")
    # Remove any potentially problematic characters
    clean_id = re.sub(r'[^\w\-_]', '', value)
    if clean_id != value:
        logger.warning(f"Paper ID '{value}' was cleaned to '{clean_id}'")
    return clean_id

def validate_abstract(value: str) -> str:
    """Validate abstract content."""
    if not isinstance(value, str):
        raise ValueError("Abstract must be a string")
    if len(value.strip()) < 10:
        raise ValueError("Abstract must be at least 10 characters long")
    if len(value) > 10000:
        raise ValueError("Abstract must be less than 10,000 characters")
    return value.strip()

# Pydantic Models for Data Validation
class Author(BaseModel):
    """Author data model with validation."""
    name: str = Field(..., min_length=1, max_length=200)
    institution: Optional[str] = Field(None, max_length=200)
    author_id: Optional[str] = Field(None, max_length=50)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Author name cannot be empty")
        return v.strip()
    
    @validator('institution')
    def validate_institution(cls, v):
        if v is not None and not v.strip():
            return None
        return v.strip() if v else None

class Keyword(BaseModel):
    """Keyword data model with validation."""
    text: str = Field(..., min_length=1, max_length=100)
    keyword_id: Optional[str] = Field(None, max_length=50)
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Keyword text cannot be empty")
        return v.strip().lower()

class Method(BaseModel):
    """Method data model with validation."""
    name: str = Field(..., min_length=1, max_length=100)
    category: Optional[str] = Field(None, max_length=50)
    method_id: Optional[str] = Field(None, max_length=50)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Method name cannot be empty")
        return v.strip()

class Paper(BaseModel):
    """Paper data model with comprehensive validation."""
    paper_id: str = Field(..., min_length=1, max_length=100)
    title: str = Field(..., min_length=1, max_length=500)
    abstract: str = Field(..., min_length=10, max_length=10000)
    year: int = Field(..., ge=1900, le=datetime.now().year + 1)
    journal: Optional[str] = Field(None, max_length=200)
    citations: int = Field(default=0, ge=0)
    authors: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    pdf_url: Optional[HttpUrl] = None
    doi: Optional[str] = Field(None, max_length=100)
    
    @validator('paper_id')
    def validate_paper_id(cls, v):
        return validate_paper_id(v)
    
    @validator('abstract')
    def validate_abstract(cls, v):
        return validate_abstract(v)
    
    @validator('authors')
    def validate_authors(cls, v):
        if not v:
            return v
        return [author.strip() for author in v if author.strip()]
    
    @validator('keywords')
    def validate_keywords(cls, v):
        if not v:
            return v
        return [keyword.strip().lower() for keyword in v if keyword.strip()]
    
    @validator('methods')
    def validate_methods(cls, v):
        if not v:
            return v
        return [method.strip() for method in v if method.strip()]
    
    @validator('doi')
    def validate_doi(cls, v):
        if v is None:
            return v
        if not re.match(r'^10\.\d{4,}/.+$', v):
            raise ValueError("Invalid DOI format")
        return v

class SearchQuery(BaseModel):
    """Search query model with validation."""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=100, ge=1, le=1000)
    max_hops: int = Field(default=2, ge=1, le=5)
    limit: int = Field(default=10, ge=1, le=100)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Search query cannot be empty")
        return v.strip()

class DatabaseConfig(BaseModel):
    """Database configuration model with validation."""
    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="password")
    
    @validator('uri')
    def validate_uri(cls, v):
        if not v.startswith(('bolt://', 'neo4j://')):
            raise ValueError("Database URI must start with 'bolt://' or 'neo4j://'")
        return v
    
    @validator('user')
    def validate_user(cls, v):
        if not v.strip():
            raise ValueError("Database user cannot be empty")
        return v.strip()
    
    @validator('password')
    def validate_password(cls, v):
        if not v.strip():
            raise ValueError("Database password cannot be empty")
        return v.strip()

class ModelConfig(BaseModel):
    """Model configuration model with validation."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    llm_model: str = Field(default="gpt-4")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    
    @validator('embedding_model')
    def validate_embedding_model(cls, v):
        valid_models = [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-MiniLM-L12-v2",
            "multi-qa-MiniLM-L6-cos-v1", "paraphrase-MiniLM-L6-v2"
        ]
        if v not in valid_models:
            logger.warning(f"Embedding model '{v}' not in known list: {valid_models}")
        return v
    
    @validator('llm_model')
    def validate_llm_model(cls, v):
        valid_models = ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]
        if v not in valid_models:
            logger.warning(f"LLM model '{v}' not in known list: {valid_models}")
        return v

# Validation Functions
def validate_environment_variables() -> Dict[str, str]:
    """Validate required environment variables."""
    required_vars = {
        "NEO4J_URI": "Database connection URI",
        "NEO4J_USER": "Database username", 
        "NEO4J_PASSWORD": "Database password",
        "OPENAI_API_KEY": "OpenAI API key"
    }
    
    missing_vars = []
    config = {}
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"{var} ({description})")
        else:
            config[var] = value
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return config

def validate_paper_data(data: Dict[str, Any]) -> Paper:
    """Validate paper data and return Paper model."""
    try:
        return Paper(**data)
    except Exception as e:
        logger.error(f"Paper validation failed: {e}")
        raise ValueError(f"Invalid paper data: {e}")

def validate_search_params(query: str, max_results: int = 100, max_hops: int = 2, limit: int = 10) -> SearchQuery:
    """Validate search parameters and return SearchQuery model."""
    try:
        return SearchQuery(
            query=query,
            max_results=max_results,
            max_hops=max_hops,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Search parameter validation failed: {e}")
        raise ValueError(f"Invalid search parameters: {e}")

def validate_database_config(uri: str = None, user: str = None, password: str = None) -> DatabaseConfig:
    """Validate database configuration and return DatabaseConfig model."""
    try:
        config_data = {}
        if uri:
            config_data["uri"] = uri
        if user:
            config_data["user"] = user
        if password:
            config_data["password"] = password
        
        return DatabaseConfig(**config_data)
    except Exception as e:
        logger.error(f"Database config validation failed: {e}")
        raise ValueError(f"Invalid database configuration: {e}")

def validate_model_config(embedding_model: str = None, llm_model: str = None, 
                         temperature: float = None, max_tokens: int = None) -> ModelConfig:
    """Validate model configuration and return ModelConfig model."""
    try:
        config_data = {}
        if embedding_model:
            config_data["embedding_model"] = embedding_model
        if llm_model:
            config_data["llm_model"] = llm_model
        if temperature is not None:
            config_data["temperature"] = temperature
        if max_tokens is not None:
            config_data["max_tokens"] = max_tokens
        
        return ModelConfig(**config_data)
    except Exception as e:
        logger.error(f"Model config validation failed: {e}")
        raise ValueError(f"Invalid model configuration: {e}")

# Type checking decorators
def type_checked(func):
    """Decorator to enable runtime type checking."""
    try:
        from typeguard import typechecked
        return typechecked(func)
    except ImportError:
        logger.warning("typeguard not available, skipping type checking")
        return func

def validate_inputs(*validators):
    """Decorator to validate function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Apply validators to arguments
            for validator_func in validators:
                try:
                    validator_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Input validation failed for {func.__name__}: {e}")
                    raise ValueError(f"Input validation failed: {e}")
            return func(*args, **kwargs)
        return wrapper
    return decorator 