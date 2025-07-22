#!/usr/bin/env python3
"""
Test script for validation and type checking features
Demonstrates the comprehensive validation system added to the codebase.
"""

import os
import sys
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from validation import (
    validate_environment_variables, validate_paper_data, validate_search_params,
    validate_database_config, validate_model_config, Paper, SearchQuery,
    DatabaseConfig, ModelConfig, ValidationError
)
from types import (
    PaperId, PaperMetadata, SearchResponse, ValidationError as TypesValidationError,
    is_paper_metadata, to_paper_metadata
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_validation():
    """Test environment variable validation."""
    print("🔧 Testing Environment Variable Validation")
    print("=" * 50)
    
    try:
        config = validate_environment_variables()
        print("✅ Environment variables validation passed")
        print(f"   Found {len(config)} required variables")
        return True
    except ValidationError as e:
        print(f"❌ Environment validation failed: {e}")
        print("   Please check your .env file")
        return False

def test_paper_validation():
    """Test paper data validation."""
    print("\n📄 Testing Paper Data Validation")
    print("=" * 50)
    
    # Valid paper data
    valid_paper = {
        "paper_id": "test_paper_123",
        "title": "A Comprehensive Study of Graph Neural Networks",
        "abstract": "This paper presents a comprehensive study of graph neural networks and their applications in various domains including computer vision, natural language processing, and scientific research.",
        "year": 2023,
        "journal": "Nature Machine Intelligence",
        "citations": 150,
        "authors": ["Dr. Sarah Johnson", "Prof. Michael Chen"],
        "keywords": ["graph neural networks", "deep learning", "machine learning"],
        "methods": ["transformer", "attention mechanism"]
    }
    
    try:
        validated_paper = validate_paper_data(valid_paper)
        print("✅ Valid paper data validation passed")
        print(f"   Paper ID: {validated_paper.paper_id}")
        print(f"   Title: {validated_paper.title}")
        print(f"   Year: {validated_paper.year}")
        print(f"   Authors: {len(validated_paper.authors)} authors")
        return True
    except ValidationError as e:
        print(f"❌ Paper validation failed: {e}")
        return False

def test_invalid_paper_validation():
    """Test invalid paper data validation."""
    print("\n🚫 Testing Invalid Paper Data Validation")
    print("=" * 50)
    
    # Invalid paper data (missing required fields)
    invalid_paper = {
        "paper_id": "",  # Empty paper ID
        "title": "",     # Empty title
        "abstract": "Too short",  # Too short abstract
        "year": 1800,    # Invalid year
        "citations": -5  # Negative citations
    }
    
    try:
        validated_paper = validate_paper_data(invalid_paper)
        print("❌ Invalid paper validation should have failed")
        return False
    except ValidationError as e:
        print("✅ Invalid paper data correctly rejected")
        print(f"   Error: {e}")
        return True

def test_search_parameter_validation():
    """Test search parameter validation."""
    print("\n🔍 Testing Search Parameter Validation")
    print("=" * 50)
    
    try:
        # Valid search parameters
        search_params = validate_search_params(
            query="deep learning",
            max_results=50,
            max_hops=3,
            limit=20
        )
        print("✅ Valid search parameters validation passed")
        print(f"   Query: {search_params.query}")
        print(f"   Max results: {search_params.max_results}")
        print(f"   Max hops: {search_params.max_hops}")
        print(f"   Limit: {search_params.limit}")
        return True
    except ValidationError as e:
        print(f"❌ Search parameter validation failed: {e}")
        return False

def test_invalid_search_parameters():
    """Test invalid search parameter validation."""
    print("\n🚫 Testing Invalid Search Parameters")
    print("=" * 50)
    
    try:
        # Invalid search parameters
        search_params = validate_search_params(
            query="",  # Empty query
            max_results=0,  # Invalid max_results
            max_hops=10,  # Too many hops
            limit=200  # Too high limit
        )
        print("❌ Invalid search parameters should have failed")
        return False
    except ValidationError as e:
        print("✅ Invalid search parameters correctly rejected")
        print(f"   Error: {e}")
        return True

def test_database_config_validation():
    """Test database configuration validation."""
    print("\n🗄️ Testing Database Configuration Validation")
    print("=" * 50)
    
    try:
        db_config = validate_database_config(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password123"
        )
        print("✅ Database configuration validation passed")
        print(f"   URI: {db_config.uri}")
        print(f"   User: {db_config.user}")
        return True
    except ValidationError as e:
        print(f"❌ Database configuration validation failed: {e}")
        return False

def test_model_config_validation():
    """Test model configuration validation."""
    print("\n🤖 Testing Model Configuration Validation")
    print("=" * 50)
    
    try:
        model_config = validate_model_config(
            embedding_model="all-MiniLM-L6-v2",
            llm_model="gpt-4",
            temperature=0.3,
            max_tokens=2000
        )
        print("✅ Model configuration validation passed")
        print(f"   Embedding model: {model_config.embedding_model}")
        print(f"   LLM model: {model_config.llm_model}")
        print(f"   Temperature: {model_config.temperature}")
        print(f"   Max tokens: {model_config.max_tokens}")
        return True
    except ValidationError as e:
        print(f"❌ Model configuration validation failed: {e}")
        return False

def test_type_guards():
    """Test type guard functions."""
    print("\n🛡️ Testing Type Guards")
    print("=" * 50)
    
    # Test valid paper metadata
    valid_metadata = {
        "paper_id": "test_123",
        "title": "Test Paper",
        "abstract": "This is a test abstract",
        "year": 2023
    }
    
    if is_paper_metadata(valid_metadata):
        print("✅ Valid paper metadata correctly identified")
    else:
        print("❌ Valid paper metadata incorrectly rejected")
        return False
    
    # Test invalid paper metadata
    invalid_metadata = {
        "paper_id": "test_123",
        # Missing required fields
    }
    
    if not is_paper_metadata(invalid_metadata):
        print("✅ Invalid paper metadata correctly rejected")
    else:
        print("❌ Invalid paper metadata incorrectly accepted")
        return False
    
    return True

def test_type_conversion():
    """Test type conversion utilities."""
    print("\n🔄 Testing Type Conversion")
    print("=" * 50)
    
    try:
        # Test conversion to PaperMetadata
        paper_data = {
            "paper_id": "convert_test_123",
            "title": "Conversion Test Paper",
            "abstract": "This paper tests type conversion functionality",
            "year": 2023,
            "journal": "Test Journal",
            "citations": 10,
            "authors": ["Test Author"],
            "keywords": ["test", "conversion"],
            "methods": ["test_method"]
        }
        
        paper_metadata = to_paper_metadata(paper_data)
        print("✅ Type conversion to PaperMetadata successful")
        print(f"   Converted paper ID: {paper_metadata['paper_id']}")
        return True
    except TypesValidationError as e:
        print(f"❌ Type conversion failed: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n📋 Testing Pydantic Models")
    print("=" * 50)
    
    try:
        # Test Paper model
        paper = Paper(
            paper_id="pydantic_test_123",
            title="Pydantic Test Paper",
            abstract="This paper tests Pydantic model validation with comprehensive validation rules and type checking.",
            year=2023,
            journal="Test Journal",
            citations=25,
            authors=["Dr. Test Author"],
            keywords=["pydantic", "validation"],
            methods=["test_method"]
        )
        print("✅ Paper Pydantic model validation passed")
        print(f"   Paper ID: {paper.paper_id}")
        print(f"   Title: {paper.title}")
        print(f"   Year: {paper.year}")
        
        # Test SearchQuery model
        search_query = SearchQuery(
            query="pydantic validation",
            max_results=100,
            max_hops=2,
            limit=10
        )
        print("✅ SearchQuery Pydantic model validation passed")
        print(f"   Query: {search_query.query}")
        print(f"   Max results: {search_query.max_results}")
        
        return True
    except ValidationError as e:
        print(f"❌ Pydantic model validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🧪 GraphRAG Validation and Type Checking Test Suite")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    tests = [
        ("Environment Variables", test_environment_validation),
        ("Paper Data Validation", test_paper_validation),
        ("Invalid Paper Data", test_invalid_paper_validation),
        ("Search Parameters", test_search_parameter_validation),
        ("Invalid Search Parameters", test_invalid_search_parameters),
        ("Database Configuration", test_database_config_validation),
        ("Model Configuration", test_model_config_validation),
        ("Type Guards", test_type_guards),
        ("Type Conversion", test_type_conversion),
        ("Pydantic Models", test_pydantic_models)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("✅ The validation and type checking system is working correctly")
    else:
        print("⚠️ Some tests failed. Please check the configuration and data.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 