#!/usr/bin/env python3
"""
run.py

UV-optimized runner for Async Podcast Generator
"""

import subprocess
import sys
import os

def main():
    """Run the podcast generator using UV"""
    print("🎙️ Starting Async Podcast Generator with UV...")
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("⚠️  No .env file found. Run setup.py first!")
        return False
    
    try:
        # Run with UV
        subprocess.check_call(["uv", "run", "python", "debate2.py"])
    except subprocess.CalledProcessError:
        print("❌ Failed to run with UV")
        return False
    except FileNotFoundError:
        print("❌ UV not found. Please install UV or run setup.py")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)