#!/usr/bin/env python3
"""
setup.py

Quick setup script for Async Podcast Generator
"""

import os
import subprocess
import sys

def main():
    print("🎙️ Async Podcast Generator - Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
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