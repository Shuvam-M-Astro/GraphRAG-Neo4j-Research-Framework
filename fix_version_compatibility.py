#!/usr/bin/env python3
"""
Version Compatibility Fix Script
Fixes compatibility issues between PyTorch and transformers versions.
"""

import subprocess
import sys
import os

def check_current_versions():
    """Check current versions of PyTorch and transformers."""
    try:
        import torch
        import transformers
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        return torch.__version__, transformers.__version__
    except ImportError as e:
        print(f"Import error: {e}")
        return None, None

def fix_version_compatibility():
    """Fix version compatibility issues."""
    print("🔧 Fixing version compatibility...")
    
    # Uninstall problematic packages
    print("Uninstalling current packages...")
    packages_to_remove = [
        "torch", "torchvision", "torchaudio", 
        "transformers", "sentence-transformers", "huggingface-hub"
    ]
    
    for package in packages_to_remove:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", package, "-y"])
    
    # Install compatible versions
    print("Installing compatible versions...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        "transformers==4.35.2",
        "sentence-transformers==2.2.2",
        "huggingface-hub==0.19.4"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Version compatibility fix successful!")
        
        # Verify the new versions
        torch_version, transformers_version = check_current_versions()
        if torch_version and transformers_version:
            print(f"✅ PyTorch: {torch_version}")
            print(f"✅ Transformers: {transformers_version}")
        
        return True
    else:
        print("❌ Version compatibility fix failed:")
        print(result.stderr)
        return False

def test_imports():
    """Test if the imports work correctly."""
    print("\n🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch imported: {torch.__version__}")
        
        import transformers
        print(f"✅ Transformers imported: {transformers.__version__}")
        
        import huggingface_hub
        print(f"✅ HuggingFace Hub imported: {huggingface_hub.__version__}")
        
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer imported")
        
        # Test basic functionality
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding = model.encode("test", convert_to_tensor=False)
        print(f"✅ SentenceTransformer test successful: {len(embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Version Compatibility Fix Tool")
    print("=" * 50)
    
    # Check current versions
    torch_version, transformers_version = check_current_versions()
    
    if torch_version and transformers_version:
        print(f"\nCurrent versions:")
        print(f"  PyTorch: {torch_version}")
        print(f"  Transformers: {transformers_version}")
        
        # Check for compatibility issues
        torch_parts = torch_version.split('.')
        transformers_parts = transformers_version.split('.')
        
        torch_major = int(torch_parts[0])
        torch_minor = int(torch_parts[1])
        transformers_major = int(transformers_parts[0])
        transformers_minor = int(transformers_parts[1])
        
        needs_fix = False
        
        if torch_major == 2 and torch_minor == 0:
            if transformers_major >= 4 and transformers_minor >= 36:
                print("⚠️  Compatibility issue detected: PyTorch 2.0.1 with transformers >=4.36.0")
                needs_fix = True
        
        if needs_fix:
            print("\n💡 This combination may cause compatibility issues.")
            response = input("Do you want to fix the version compatibility? (y/n): ").lower().strip()
            
            if response == 'y':
                if fix_version_compatibility():
                    print("\n🎉 Version compatibility fix completed!")
                    print("Testing imports...")
                    if test_imports():
                        print("✅ All tests passed! Your setup should work correctly.")
                    else:
                        print("❌ Some tests failed. Please check the error messages above.")
                else:
                    print("\n❌ Fix failed. Please try manually:")
                    print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 transformers==4.35.2 sentence-transformers==2.2.2")
            else:
                print("Skipping fix. You may continue to experience compatibility issues.")
        else:
            print("✅ Version compatibility looks good!")
            if test_imports():
                print("✅ All imports working correctly!")
            else:
                print("❌ Import test failed. Consider running the fix anyway.")
    else:
        print("Installing compatible versions...")
        if fix_version_compatibility():
            print("✅ Installation successful!")
            if test_imports():
                print("✅ All tests passed!")
        else:
            print("❌ Installation failed!")

if __name__ == "__main__":
    main() 