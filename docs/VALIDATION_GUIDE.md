# GraphRAG Validation and Type Checking Guide

This guide documents the comprehensive validation and static type checking system implemented in the GraphRAG Neo4j Research Framework.

## Overview

The validation system provides:
- **Input Validation**: Comprehensive validation of all function inputs
- **Data Validation**: Pydantic models for data structure validation
- **Type Checking**: Static type checking with mypy and runtime type checking with typeguard
- **Error Handling**: Custom exception types for different validation failures
- **Performance Monitoring**: Built-in performance tracking and statistics

## Key Components

### 1. Validation Module (`src/validation.py`)

#### Pydantic Models
- `Paper`: Validates paper data with comprehensive rules
- `SearchQuery`: Validates search parameters
- `DatabaseConfig`: Validates database connection settings
- `ModelConfig`: Validates model configuration parameters

#### Validation Functions
- `validate_environment_variables()`: Checks required environment variables
- `validate_paper_data()`: Validates paper data against Pydantic model
- `validate_search_params()`: Validates search parameters
- `validate_database_config()`: Validates database configuration
- `validate_model_config()`: Validates model configuration

#### Decorators
- `@type_checked`: Enables runtime type checking
- `@validate_inputs`: Validates function inputs

### 2. Types Module (`src/types.py`)

#### Type Definitions
- Custom type aliases (`PaperId`, `AuthorId`, etc.)
- TypedDict classes for structured data
- Protocol classes for callable types
- Custom exception types

#### Type Guards
- `is_paper_metadata()`: Checks if object is valid paper metadata
- `is_search_response()`: Checks if object is valid search response
- `is_valid_paper_id()`: Validates paper ID format
- `is_valid_year()`: Validates year for research papers
- `is_valid_citation_count()`: Validates citation count

#### Type Conversion
- `to_paper_metadata()`: Converts dictionary to PaperMetadata
- `to_search_response()`: Converts dictionary to SearchResponse

## Usage Examples

### Basic Validation

```python
from src.validation import validate_paper_data, Paper
from src.types import ValidationError

# Valid paper data
paper_data = {
    "paper_id": "test_123",
    "title": "Test Paper",
    "abstract": "This is a comprehensive test abstract.",
    "year": 2023,
    "journal": "Test Journal",
    "citations": 10,
    "authors": ["Test Author"],
    "keywords": ["test"],
    "methods": ["test_method"]
}

try:
    validated_paper = validate_paper_data(paper_data)
    print(f"Valid paper: {validated_paper.title}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Search Parameter Validation

```python
from src.validation import validate_search_params

try:
    search_params = validate_search_params(
        query="deep learning",
        max_results=50,
        max_hops=3,
        limit=20
    )
    print(f"Valid search: {search_params.query}")
except ValidationError as e:
    print(f"Search validation failed: {e}")
```

### Database Configuration Validation

```python
from src.validation import validate_database_config

try:
    db_config = validate_database_config(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password123"
    )
    print(f"Valid database config: {db_config.uri}")
except ValidationError as e:
    print(f"Database config validation failed: {e}")
```

### Using Decorators

```python
from src.validation import type_checked, validate_inputs
from src.types import PaperMetadata

@type_checked
@validate_inputs
def process_paper(paper: PaperMetadata, analysis_type: str) -> str:
    """Process a paper with validation."""
    if analysis_type not in ["summary", "detailed"]:
        raise ValueError("Invalid analysis type")
    
    return f"Processed {paper.title} with {analysis_type} analysis"
```

## Validation Rules

### Paper Validation
- **paper_id**: Non-empty string, max 100 characters
- **title**: Non-empty string, max 500 characters
- **abstract**: 10-10,000 characters
- **year**: 1900 to current year + 1
- **citations**: Non-negative integer
- **authors**: List of non-empty strings
- **keywords**: List of lowercase strings
- **methods**: List of non-empty strings
- **doi**: Valid DOI format (if provided)

### Search Parameter Validation
- **query**: Non-empty string, max 500 characters
- **max_results**: 1-1000
- **max_hops**: 1-5
- **limit**: 1-100

### Database Configuration Validation
- **uri**: Must start with 'bolt://' or 'neo4j://'
- **user**: Non-empty string
- **password**: Non-empty string

### Model Configuration Validation
- **embedding_model**: Known sentence transformer model
- **llm_model**: Known OpenAI model
- **temperature**: 0.0-2.0
- **max_tokens**: 1-8000

## Error Types

### ValidationError
Raised when data validation fails:
```python
from src.validation import ValidationError

try:
    # Validation code
    pass
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Custom Exception Types
- `ConfigurationError`: Invalid configuration
- `DatabaseError`: Database operation failures
- `SearchError`: Search operation failures
- `ModelError`: Model operation failures

## Type Checking

### Static Type Checking with mypy

Run static type checking:
```bash
mypy src/
```

Configuration in `mypy.ini`:
- Strict checking for source code
- Ignore missing imports for external libraries
- Relaxed checking for test files and Streamlit apps

### Runtime Type Checking

Enable runtime type checking with the `@type_checked` decorator:
```python
from src.validation import type_checked

@type_checked
def process_data(data: List[Dict[str, Any]]) -> List[PaperMetadata]:
    # Function will be type-checked at runtime
    pass
```

## Performance Monitoring

### Built-in Statistics

The validation system includes performance tracking:

```python
# GraphRetriever performance stats
retriever = GraphRetriever()
stats = retriever.get_performance_stats()
print(f"Average search time: {stats['average_search_time']:.2f}s")

# ArXivIngestion performance stats
ingestion = ArXivIngestion()
print(f"Success rate: {ingestion.ingestion_stats['successful_stores'] / ingestion.ingestion_stats['total_papers_processed']:.2%}")
```

### Index Integrity Validation

```python
# Validate vector index integrity
if not retriever.validate_index_integrity():
    print("Index corrupted, rebuilding...")
    retriever.rebuild_index_if_needed()
```

## Testing

### Run Validation Tests

```bash
python test_validation.py
```

This will test:
- Environment variable validation
- Paper data validation (valid and invalid cases)
- Search parameter validation
- Database configuration validation
- Model configuration validation
- Type guards and conversion utilities
- Pydantic model validation

### Expected Output

```
🧪 GraphRAG Validation and Type Checking Test Suite
============================================================
🔧 Testing Environment Variable Validation
==================================================
✅ Environment variables validation passed
   Found 4 required variables

📄 Testing Paper Data Validation
==================================================
✅ Valid paper data validation passed
   Paper ID: test_paper_123
   Title: A Comprehensive Study of Graph Neural Networks
   Year: 2023
   Authors: 2 authors

...

📊 Test Results: 10/10 tests passed
🎉 All validation tests passed!
✅ The validation and type checking system is working correctly
```

## Integration with Existing Code

### GraphRetriever Enhancements

The `GraphRetriever` class now includes:
- Input validation for all public methods
- Comprehensive error handling
- Performance tracking
- Index integrity validation
- Type-safe return values

### ArXivIngestion Enhancements

The `ArXivIngestion` class now includes:
- Paper data validation before storage
- Input parameter validation
- Performance statistics
- Comprehensive error handling

## Best Practices

### 1. Always Validate Inputs
```python
@validate_retriever_inputs
def search_method(self, query: str, limit: int = 10) -> List[PaperMetadata]:
    # Method is automatically validated
    pass
```

### 2. Use Type Annotations
```python
def process_papers(papers: List[PaperMetadata]) -> SearchResponse:
    # Clear type annotations help with validation
    pass
```

### 3. Handle Validation Errors
```python
try:
    result = validate_paper_data(paper_dict)
except ValidationError as e:
    logger.error(f"Paper validation failed: {e}")
    # Handle gracefully
```

### 4. Monitor Performance
```python
# Check performance regularly
stats = retriever.get_performance_stats()
if stats['average_search_time'] > 5.0:
    logger.warning("Search performance degraded")
```

## Configuration

### Environment Variables

Required environment variables (validated on startup):
- `NEO4J_URI`: Database connection URI
- `NEO4J_USER`: Database username
- `NEO4J_PASSWORD`: Database password
- `OPENAI_API_KEY`: OpenAI API key

### mypy Configuration

The `mypy.ini` file configures static type checking:
- Strict checking for source code
- Relaxed checking for external libraries
- Custom rules for different modules

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure validation modules are in the Python path
2. **Type Errors**: Check mypy configuration and type annotations
3. **Validation Failures**: Review validation rules and data format
4. **Performance Issues**: Monitor statistics and rebuild indexes if needed

### Debug Mode

Enable debug logging to see validation details:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements:
- Schema validation for Neo4j queries
- Advanced NLP-based content validation
- Machine learning-based anomaly detection
- Real-time validation monitoring
- Integration with external validation services

## Conclusion

The validation and type checking system provides a robust foundation for data integrity and code quality. By following the patterns and best practices outlined in this guide, you can ensure reliable operation of the GraphRAG system while maintaining high code quality standards. 