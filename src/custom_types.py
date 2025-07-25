"""
Type definitions for GraphRAG Neo4j Research Framework
Provides comprehensive type hints for better code quality and IDE support.
"""

from typing import (
    List, Dict, Any, Optional, Tuple, Union, Callable, 
    Protocol, TypedDict, Literal, NewType
)
from datetime import datetime, date
from pathlib import Path
import numpy as np

# Custom type aliases for better readability
PaperId = NewType('PaperId', str)
AuthorId = NewType('AuthorId', str)
MethodId = NewType('MethodId', str)
KeywordId = NewType('KeywordId', str)

# Vector types
EmbeddingVector = np.ndarray  # Type: np.ndarray[Any, np.dtype[np.float32]]
EmbeddingMatrix = np.ndarray  # Type: np.ndarray[Any, np.dtype[np.float32]]

# Search result types
SearchScore = float
SearchResult = Tuple[SearchScore, int]  # (score, document_index)

# Database types
Neo4jRecord = Dict[str, Any]
Neo4jResult = List[Neo4jRecord]

# API Response types
class ApiResponse(TypedDict):
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    timestamp: str

class SearchResponse(TypedDict):
    query: str
    results: List[Dict[str, Any]]
    total_found: int
    search_time: float
    metadata: Dict[str, Any]

class ResearchAnalysisResponse(TypedDict):
    topic: str
    summary: str
    papers_analyzed: int
    key_findings: List[str]
    research_gaps: List[str]
    recommendations: List[str]
    generated_at: str

# Paper-related types
class PaperMetadata(TypedDict):
    paper_id: PaperId
    title: str
    abstract: str
    year: int
    journal: Optional[str]
    citations: int
    authors: List[str]
    keywords: List[str]
    methods: List[str]

class PaperWithScore(TypedDict):
    paper_id: PaperId
    title: str
    abstract: str
    year: int
    journal: Optional[str]
    citations: int
    authors: List[str]
    keywords: List[str]
    methods: List[str]
    vector_score: float
    graph_score: float
    hybrid_score: float

# Graph types
class GraphNode(TypedDict):
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class GraphRelationship(TypedDict):
    start_node: str
    end_node: str
    type: str
    properties: Dict[str, Any]

class GraphPath(TypedDict):
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int

# Configuration types
class DatabaseSettings(TypedDict):
    uri: str
    user: str
    password: str
    max_connection_pool_size: int
    connection_timeout: int

class ModelSettings(TypedDict):
    embedding_model: str
    llm_model: str
    temperature: float
    max_tokens: int
    top_k: int
    top_p: float

class SearchSettings(TypedDict):
    max_results: int
    max_hops: int
    similarity_threshold: float
    enable_hybrid_search: bool
    alpha: float  # Weight for vector vs graph search

# Validation types
ValidationResult = Tuple[bool, Optional[str]]  # (is_valid, error_message)

# Callback types
class ProgressCallback(Protocol):
    def __call__(self, current: int, total: int, message: str) -> None:
        ...

class ErrorCallback(Protocol):
    def __call__(self, error: Exception, context: str) -> None:
        ...

# Analysis types
AnalysisType = Literal[
    "comprehensive", 
    "summary", 
    "gap_analysis", 
    "methodology_tracking",
    "collaboration_analysis",
    "literature_review"
]

SearchType = Literal[
    "vector", 
    "graph", 
    "hybrid", 
    "multi_hop"
]

# File types
FilePath = Union[str, Path]
FileContent = Union[str, bytes]

# Network types
class CollaborationNetwork(TypedDict):
    authors: List[str]
    collaborations: List[Tuple[str, str]]
    institutions: Dict[str, str]
    collaboration_strength: Dict[Tuple[str, str], int]

class CitationNetwork(TypedDict):
    papers: List[PaperId]
    citations: List[Tuple[PaperId, PaperId]]
    citation_counts: Dict[PaperId, int]

# Method evolution types
class MethodEvolution(TypedDict):
    method_name: str
    timeline: List[Tuple[int, int]]  # (year, usage_count)
    disciplines: List[str]
    applications: List[str]
    evolution_pattern: str

# Research gap types
class ResearchGap(TypedDict):
    gap_type: str
    description: str
    severity: Literal["low", "medium", "high"]
    opportunities: List[str]
    related_papers: List[PaperId]

# Validation function types
class Validator(Protocol):
    def __call__(self, value: Any) -> ValidationResult:
        ...

class DataValidator(Protocol):
    def __call__(self, data: Dict[str, Any]) -> ValidationResult:
        ...

# Cache types
class CacheEntry(TypedDict):
    key: str
    value: Any
    timestamp: float
    ttl: int

# Logging types
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

class LogEntry(TypedDict):
    level: LogLevel
    message: str
    timestamp: datetime
    module: str
    function: str
    line_number: int
    extra_data: Optional[Dict[str, Any]]

# Performance monitoring types
class PerformanceMetrics(TypedDict):
    function_name: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    timestamp: datetime

# Error types
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass

class SearchError(Exception):
    """Raised when search operations fail."""
    pass

class ModelError(Exception):
    """Raised when model operations fail."""
    pass

# Type guards for runtime type checking
def is_paper_metadata(obj: Any) -> bool:
    """Check if object is a valid PaperMetadata."""
    if not isinstance(obj, dict):
        return False
    
    required_fields = {'paper_id', 'title', 'abstract', 'year'}
    return all(field in obj for field in required_fields)

def is_search_response(obj: Any) -> bool:
    """Check if object is a valid SearchResponse."""
    if not isinstance(obj, dict):
        return False
    
    required_fields = {'query', 'results', 'total_found'}
    return all(field in obj for field in required_fields)

def is_valid_paper_id(paper_id: Any) -> bool:
    """Check if paper_id is valid."""
    if not isinstance(paper_id, str):
        return False
    return len(paper_id.strip()) > 0 and len(paper_id) <= 100

def is_valid_year(year: Any) -> bool:
    """Check if year is valid for research papers."""
    if not isinstance(year, int):
        return False
    return 1900 <= year <= datetime.now().year + 1

def is_valid_citation_count(citations: Any) -> bool:
    """Check if citation count is valid."""
    if not isinstance(citations, int):
        return False
    return citations >= 0

# Type conversion utilities
def to_paper_metadata(data: Dict[str, Any]) -> PaperMetadata:
    """Convert dictionary to PaperMetadata with validation."""
    if not is_paper_metadata(data):
        raise ValidationError("Invalid paper metadata format")
    
    return PaperMetadata(
        paper_id=data['paper_id'],
        title=data['title'],
        abstract=data['abstract'],
        year=data['year'],
        journal=data.get('journal'),
        citations=data.get('citations', 0),
        authors=data.get('authors', []),
        keywords=data.get('keywords', []),
        methods=data.get('methods', [])
    )

def to_search_response(data: Dict[str, Any]) -> SearchResponse:
    """Convert dictionary to SearchResponse with validation."""
    if not is_search_response(data):
        raise ValidationError("Invalid search response format")
    
    return SearchResponse(
        query=data['query'],
        results=data['results'],
        total_found=data['total_found'],
        search_time=data.get('search_time', 0.0),
        metadata=data.get('metadata', {})
    ) 