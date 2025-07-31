#!/usr/bin/env python3
"""
Automatic PyTorch Downgrade Script
Downgrades PyTorch to a version that doesn't have meta tensor issues.
"""

import subprocess
import sys
import os

def check_current_version():
    """Check current PyTorch version."""
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch version: {version}")
        return version
    except ImportError:
        print("PyTorch not installed")
        return None

def downgrade_pytorch():
    """Downgrade PyTorch to a stable version."""
    print("🚀 Downgrading PyTorch to version 2.0.1...")
    
    # Uninstall current PyTorch
    print("Uninstalling current PyTorch...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    
    # Install stable version
    print("Installing PyTorch 2.0.1...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.0.1",
        "torchvision==0.15.2", 
        "torchaudio==2.0.2"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ PyTorch downgrade successful!")
        
        # Verify the new version
        new_version = check_current_version()
        if new_version:
            print(f"✅ New PyTorch version: {new_version}")
        
        return True
    else:
        print("❌ PyTorch downgrade failed:")
        print(result.stderr)
        return False

def main():
    """Main function."""
    print("🔧 PyTorch Downgrade Tool")
    print("=" * 40)
    
    current_version = check_current_version()
    
    if current_version:
        # Check if downgrade is needed
        version_parts = current_version.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        
        if major >= 2 and minor >= 1:
            print("⚠️  PyTorch 2.1+ detected - this version has meta tensor issues")
            print("💡 Downgrading to PyTorch 2.0.1 will resolve the embedding model issues")
            
            response = input("\nDo you want to downgrade PyTorch? (y/n): ").lower().strip()
            
            if response == 'y':
                if downgrade_pytorch():
                    print("\n🎉 Downgrade completed successfully!")
                    print("Please restart your application and try again.")
                else:
                    print("\n❌ Downgrade failed. Please try manually:")
                    print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2")
            else:
                print("Skipping downgrade. You may continue to experience meta tensor issues.")
        else:
            print("✅ Your PyTorch version should work fine!")
            print("If you're still having issues, try the embedding model test:")
            print("python test_embedding_fix.py")
    else:
        print("Installing PyTorch 2.0.1...")
        if downgrade_pytorch():
            print("✅ PyTorch installation successful!")
        else:
            print("❌ PyTorch installation failed!")

if __name__ == "__main__":
    main() 