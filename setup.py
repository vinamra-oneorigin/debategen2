#!/usr/bin/env python3
"""
setup.py

Quick setup script for Async Podcast Generator using UV
"""

import os
import subprocess
import sys
import shutil

def check_uv_installed():
    """Check if UV is installed and available"""
    try:
        subprocess.check_call(["uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_uv():
    """Install UV if not present"""
    print("📦 UV not found. Installing UV...")
    try:
        # Try to install via pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
        print("✅ UV installed successfully via pip")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install UV via pip")
        print("Please install UV manually: https://docs.astral.sh/uv/getting-started/installation/")
        return False

def main():
    print("🎙️ Async Podcast Generator - Setup (UV)")
    print("=" * 42)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required (due to Gradio 5.0+ dependency)")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check and install UV if needed
    if not check_uv_installed():
        if not install_uv():
            return False
    else:
        print("✅ UV package manager detected")
    
    # Install dependencies using UV
    print("\n📦 Installing dependencies with UV...")
    try:
        subprocess.check_call(["uv", "sync"])
        print("✅ Dependencies installed successfully with UV")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies with UV")
        print("Falling back to pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully with pip")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
    
    # Check for .env file
    if not os.path.exists(".env"):
        print("\n⚙️ Setting up environment file...")
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from template")
            print("\n⚠️  IMPORTANT: Edit .env file with your API keys before running!")
            print("   - OpenAI API Key")
            print("   - ElevenLabs API Key") 
            print("   - Voice IDs from ElevenLabs")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    
    # Create output directory
    os.makedirs("output_audio", exist_ok=True)
    print("✅ Output directory created")
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python debate2.py")
    print("3. Open: http://localhost:7900")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)