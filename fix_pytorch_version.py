#!/usr/bin/env python3
"""
PyTorch Version Compatibility Fix Script
Helps users resolve meta tensor issues by downgrading PyTorch if needed.
"""

import subprocess
import sys
import os

def check_pytorch_version():
    """Check current PyTorch version."""
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch version: {version}")
        
        # Parse version
        parts = version.split('.')
        major = int(parts[0])
        minor = int(parts[1])
        
        if major >= 2 and minor >= 1:
            print("⚠️  PyTorch 2.1+ detected - may have meta tensor issues")
            return True
        else:
            print("✅ PyTorch version should be compatible")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def suggest_downgrade():
    """Suggest PyTorch downgrade commands."""
    print("\n🔧 PyTorch Downgrade Options:")
    print("=" * 50)
    
    print("\nOption 1: Downgrade to PyTorch 2.0.1 (Recommended)")
    print("pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2")
    
    print("\nOption 2: Downgrade to PyTorch 1.13.1 (Most Stable)")
    print("pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1")
    
    print("\nOption 3: Install CPU-only version (Lightweight)")
    print("pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu -f https://download.pytorch.org/whl/torch_stable.html")

def run_downgrade_command():
    """Run the downgrade command."""
    print("\n🚀 Attempting PyTorch downgrade...")
    
    try:
        # Use PyTorch 2.0.1 which is more stable
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PyTorch downgrade successful!")
            print("Please restart your application.")
        else:
            print("❌ PyTorch downgrade failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error during downgrade: {e}")

def main():
    """Main function."""
    print("🔬 PyTorch Version Compatibility Checker")
    print("=" * 50)
    
    # Check current version
    needs_downgrade = check_pytorch_version()
    
    if needs_downgrade:
        print("\n💡 Meta tensor issues detected. This can cause embedding model initialization failures.")
        
        # Ask user what they want to do
        print("\nOptions:")
        print("1. Show downgrade commands")
        print("2. Attempt automatic downgrade")
        print("3. Continue with current version (may have issues)")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            suggest_downgrade()
        elif choice == "2":
            run_downgrade_command()
        elif choice == "3":
            print("⚠️  Continuing with current version. You may encounter meta tensor errors.")
        else:
            print("Invalid choice. Showing downgrade commands:")
            suggest_downgrade()
    else:
        print("\n✅ Your PyTorch version should work fine!")
        print("If you're still having issues, try the embedding model test:")
        print("python test_embedding_fix.py")

if __name__ == "__main__":
    main() 