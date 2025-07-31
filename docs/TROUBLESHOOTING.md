# Troubleshooting Guide

## Meta Tensor Error Resolution

If you're encountering the error:
```
Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
```

This is a known issue with PyTorch 2.1+ and SentenceTransformer. Here are the solutions:

### 🔧 Quick Fix: Version Compatibility

**Option 1: Automatic Fix**
```bash
python fix_version_compatibility.py
```

**Option 2: Manual Fix**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 transformers==4.35.2 sentence-transformers==2.2.2
```

**Option 3: PyTorch Downgrade Only**
```bash
python downgrade_pytorch.py
```

### 🧪 Test Your Setup

After making changes, test your embedding model:
```bash
python test_embedding_fix.py
```

### 🔍 Diagnostic Tools

**Check PyTorch Version:**
```bash
python fix_pytorch_version.py
```

**System Diagnostics:**
- Use the "System Diagnostics" page in the web app
- Check PyTorch version, CUDA availability, and model initialization

### 🚀 Enhanced Initialization

The system now includes multiple fallback methods:

1. **Minimal initialization** without device specification
2. **CPU-only environment** with torch.no_grad()
3. **Alternative model** (paraphrase-MiniLM-L3-v2)
4. **Manual HuggingFace loading** with custom wrapper
5. **Dummy model** for testing (random embeddings)

### 📋 Common Solutions

#### **Solution 1: PyTorch Downgrade (Recommended)**
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install stable version
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

#### **Solution 2: Environment Variables**
Set these environment variables before running:
```bash
export TOKENIZERS_PARALLELISM=false
export PYTORCH_DISABLE_META_TENSOR=1
```

#### **Solution 3: Virtual Environment**
Create a fresh virtual environment:
```bash
python -m venv graphrag_env
source graphrag_env/bin/activate  # On Windows: graphrag_env\Scripts\activate
pip install -r requirements.txt
```

### 🔧 Advanced Troubleshooting

#### **Check PyTorch Installation**
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

#### **Test SentenceTransformer Directly**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("test")
print(f"Embedding dimension: {len(embedding)}")
```

#### **Check System Resources**
- Ensure sufficient RAM (at least 4GB free)
- Check disk space for model downloads
- Verify internet connection for model downloads

### 🐛 Known Issues

#### **Issue 1: Meta Tensor Error**
- **Cause**: PyTorch 2.1+ meta tensor implementation
- **Solution**: Downgrade to PyTorch 2.0.1

#### **Issue 2: Version Compatibility Error**
- **Cause**: transformers >=4.36.0 requires PyTorch 2.1+ but we're using 2.0.1
- **Solution**: Use compatible versions (transformers==4.35.2 with torch==2.0.1)

#### **Issue 3: CUDA Memory Issues**
- **Cause**: GPU memory insufficient or CUDA version mismatch
- **Solution**: Force CPU-only mode or upgrade GPU drivers

#### **Issue 4: Model Download Failures**
- **Cause**: Network issues or HuggingFace server problems
- **Solution**: Check internet connection, try different network

### 📞 Getting Help

If you're still experiencing issues:

1. **Run the diagnostic tests:**
   ```bash
   python test_embedding_fix.py
   python fix_pytorch_version.py
   ```

2. **Check the logs** for specific error messages

3. **Try the web app diagnostics** page

4. **Report the issue** with:
   - PyTorch version
   - Error message
   - System specifications
   - Steps to reproduce

### 🎯 Success Indicators

When everything is working correctly, you should see:
- ✅ "Embedding model initialized successfully"
- ✅ "Model test successful, embedding dimension: 384"
- ✅ "All tests passed! Embedding model should work correctly"

### 🔄 Recovery Steps

If the system fails to initialize:

1. **Clear cache and restart**
2. **Downgrade PyTorch** using the provided script
3. **Test with dummy model** (random embeddings for testing)
4. **Check system resources** and network connectivity
5. **Try alternative models** if the default fails

### 📊 Performance Notes

- **CPU-only mode**: Slower but more reliable
- **GPU mode**: Faster but may have compatibility issues
- **Dummy model**: For testing only, not for production use
- **Alternative models**: May have different performance characteristics

### 🔒 Security Considerations

- **Model downloads**: Verify HTTPS connections
- **Environment variables**: Don't expose sensitive data
- **Dummy model**: Only for testing, not for real embeddings
- **Version pinning**: Ensures reproducible builds 