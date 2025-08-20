#!/usr/bin/env python3
"""
Setup script to install all dependencies for the Audio Transcriber and Summarizer.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("   Requires Python 3.8 or higher")
        return False


def install_dependencies():
    """Install all required dependencies."""
    print("\n📦 Installing Dependencies...")
    
    # Get the Python executable from the virtual environment
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    
    if venv_python.exists():
        python_cmd = str(venv_python)
        print(f"🎯 Using virtual environment: {python_cmd}")
    else:
        python_cmd = sys.executable
        print(f"🎯 Using system Python: {python_cmd}")
    
    # Core dependencies
    dependencies = [
        "openai-whisper",
        "torch",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "scipy>=1.10.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    ]
    
    # Install core dependencies
    for dep in dependencies:
        success = run_command(
            f'"{python_cmd}" -m pip install "{dep}"',
            f"Installing {dep}"
        )
        if not success:
            print(f"⚠️  Failed to install {dep}, continuing...")
    
    # Special handling for moviepy (often problematic)
    print("\n🎬 Installing MoviePy for video processing...")
    
    # Try different approaches for moviepy
    moviepy_approaches = [
        f'"{python_cmd}" -m pip install moviepy',
        f'"{python_cmd}" -m pip install moviepy --no-cache-dir',
        f'"{python_cmd}" -m pip install imageio-ffmpeg moviepy',
    ]
    
    moviepy_installed = False
    for approach in moviepy_approaches:
        if run_command(approach, "Installing MoviePy"):
            moviepy_installed = True
            break
    
    if not moviepy_installed:
        print("⚠️  MoviePy installation failed. Video processing may not work.")
        print("   You can try installing manually: pip install moviepy")
    
    # Test imports
    print("\n🧪 Testing imports...")
    test_imports = [
        ("whisper", "OpenAI Whisper"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("moviepy.editor", "MoviePy"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile")
    ]
    
    for module, name in test_imports:
        try:
            result = subprocess.run(
                f'"{python_cmd}" -c "import {module}; print(\'✅ {name} imported successfully\')"',
                shell=True, check=True, capture_output=True, text=True
            )
            print(result.stdout.strip())
        except subprocess.CalledProcessError:
            print(f"❌ {name} import failed")


def main():
    """Main setup function."""
    print("🚀 Audio Transcriber and Summarizer - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    install_dependencies()
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Test the modules: python test_modules.py")
    print("2. Run the main script: python main.py webcams.webm")
    print("3. Or use batch processing: python main.py video.webm --batch-duration 180")


if __name__ == "__main__":
    main()
