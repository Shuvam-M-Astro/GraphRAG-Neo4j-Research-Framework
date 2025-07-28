#!/usr/bin/env python3
"""
Test script to verify embedding model initialization fix.
"""

import os
import sys
import logging

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from graph_rag.graph_retriever import safe_initialize_sentence_transformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_embedding_initialization():
    """Test the safe embedding model initialization."""
    try:
        logger.info("Testing embedding model initialization...")
        
        # Test with the default model
        model = safe_initialize_sentence_transformer("all-MiniLM-L6-v2")
        
        # Test that we can get embeddings
        test_text = "This is a test sentence."
        embeddings = model.encode(test_text)
        
        logger.info(f"✅ Embedding model initialized successfully!")
        logger.info(f"   - Model dimension: {model.get_sentence_embedding_dimension()}")
        logger.info(f"   - Test embedding shape: {embeddings.shape}")
        logger.info(f"   - Test embedding successful: {len(embeddings) > 0}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding model initialization failed: {e}")
        return False

def test_pytorch_environment():
    """Test PyTorch environment and compatibility."""
    try:
        import torch
        import transformers
        
        logger.info("🔍 PyTorch Environment Test")
        logger.info(f"   - PyTorch version: {torch.__version__}")
        logger.info(f"   - Transformers version: {transformers.__version__}")
        logger.info(f"   - CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"   - CUDA version: {torch.version.cuda}")
            logger.info(f"   - GPU count: {torch.cuda.device_count()}")
        
        # Test basic tensor operations
        test_tensor = torch.tensor([1.0, 2.0, 3.0])
        logger.info(f"   - Basic tensor test: {test_tensor}")
        
        # Test device operations
        cpu_tensor = test_tensor.cpu()
        logger.info(f"   - CPU tensor test: {cpu_tensor}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ PyTorch environment test failed: {e}")
        return False

def test_sentence_transformer_import():
    """Test SentenceTransformer import and basic functionality."""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info("🔍 SentenceTransformer Import Test")
        logger.info("   - Import successful")
        
        # Test basic initialization without device
        logger.info("   - Testing basic initialization...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("   - Basic initialization successful")
        
        # Test encoding
        test_embedding = model.encode("test", convert_to_tensor=False)
        logger.info(f"   - Encoding test successful, dimension: {len(test_embedding)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SentenceTransformer import test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Starting Embedding Model Tests")
    logger.info("=" * 50)
    
    # Run all tests
    tests = [
        ("PyTorch Environment", test_pytorch_environment),
        ("SentenceTransformer Import", test_sentence_transformer_import),
        ("Embedding Model Initialization", test_embedding_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        result = test_func()
        results.append((test_name, result))
        logger.info(f"{'✅ PASSED' if result else '❌ FAILED'}: {test_name}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"   {status}: {test_name}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Embedding model should work correctly.")
    else:
        logger.error("⚠️  Some tests failed. Check the logs above for details.")
    
    sys.exit(0 if passed == total else 1) 