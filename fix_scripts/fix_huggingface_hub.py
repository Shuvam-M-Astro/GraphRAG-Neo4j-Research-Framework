#!/usr/bin/env python3
"""
HuggingFace Hub Compatibility Fix
Fixes the 'cached_download' import error.
"""

import subprocess
import sys

def fix_huggingface_hub():
    """Fix huggingface_hub compatibility issues."""
    print("🔧 Fixing HuggingFace Hub compatibility...")
    
    # Uninstall current huggingface_hub
    print("Uninstalling current huggingface_hub...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "huggingface-hub", "-y"])
    
    # Install compatible version
    print("Installing compatible huggingface_hub...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "huggingface-hub==0.19.4"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ HuggingFace Hub fix successful!")
        return True
    else:
        print("❌ HuggingFace Hub fix failed:")
        print(result.stderr)
        return False

def test_huggingface_hub():
    """Test if huggingface_hub works correctly."""
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub imported: {huggingface_hub.__version__}")
        
        # Test if cached_download is available
        try:
            from huggingface_hub import cached_download
            print("✅ cached_download function available")
            return True
        except ImportError:
            print("❌ cached_download function not available")
            return False
            
    except Exception as e:
        print(f"❌ HuggingFace Hub test failed: {e}")
        return False

def main():
    """Main function."""
    print("🔧 HuggingFace Hub Compatibility Fix")
    print("=" * 40)
    
    if fix_huggingface_hub():
        print("\n🧪 Testing HuggingFace Hub...")
        if test_huggingface_hub():
            print("✅ HuggingFace Hub compatibility fix successful!")
            print("You can now run: python test_embedding_fix.py")
        else:
            print("❌ HuggingFace Hub test failed. Try the full version compatibility fix:")
            print("python fix_version_compatibility.py")
    else:
        print("❌ Fix failed. Try the full version compatibility fix:")
        print("python fix_version_compatibility.py")

if __name__ == "__main__":
    main() 