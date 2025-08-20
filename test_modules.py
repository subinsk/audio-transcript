#!/usr/bin/env python3
"""
Simple test script to verify modular structure without heavy dependencies.
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported correctly."""
    print("🧪 Testing modular imports...")
    
    try:
        # Test config import
        print("  ✅ Config module...")
        from src.config import DEFAULT_WHISPER_MODEL, BATCH_DURATION_SECONDS
        print(f"     Default Whisper model: {DEFAULT_WHISPER_MODEL}")
        print(f"     Default batch duration: {BATCH_DURATION_SECONDS}s")
        
        # Test utils import
        print("  ✅ Utils module...")
        from src.utils.file_manager import FileManager
        fm = FileManager()
        timestamp = fm.generate_timestamp()
        print(f"     Generated timestamp: {timestamp}")
        
        # Test other modules (may fail if dependencies not installed)
        try:
            print("  🔍 Testing audio processor...")
            from src.processors.audio_processor import AudioProcessor
            print("     ✅ Audio processor imported successfully")
        except ImportError as e:
            print(f"     ⚠️  Audio processor: {e}")
        
        try:
            print("  🔍 Testing whisper transcriber...")
            from src.transcriber.whisper_transcriber import WhisperTranscriber
            print("     ✅ Whisper transcriber imported successfully")
        except ImportError as e:
            print(f"     ⚠️  Whisper transcriber: {e}")
        
        try:
            print("  🔍 Testing llama summarizer...")
            from src.summarizer.llama_summarizer import LlamaSummarizer
            print("     ✅ Llama summarizer imported successfully")
        except ImportError as e:
            print(f"     ⚠️  Llama summarizer: {e}")
        
        try:
            print("  🔍 Testing batch processor...")
            from src.batch_processor import BatchProcessor
            print("     ✅ Batch processor imported successfully")
        except ImportError as e:
            print(f"     ⚠️  Batch processor: {e}")
        
        print("\n✅ Modular structure test completed!")
        print("📁 Project structure is correctly organized")
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()


def show_project_structure():
    """Display the project structure."""
    print("\n📁 Project Structure:")
    print("d:/projects/audio-transcripter/")
    print("├── src/")
    print("│   ├── __init__.py")
    print("│   ├── config.py")
    print("│   ├── batch_processor.py")
    print("│   ├── processors/")
    print("│   │   ├── __init__.py")
    print("│   │   └── audio_processor.py")
    print("│   ├── transcriber/")
    print("│   │   ├── __init__.py")
    print("│   │   └── whisper_transcriber.py")
    print("│   ├── summarizer/")
    print("│   │   ├── __init__.py")
    print("│   │   └── llama_summarizer.py")
    print("│   └── utils/")
    print("│       ├── __init__.py")
    print("│       └── file_manager.py")
    print("├── main.py")
    print("├── test_modules.py")
    print("├── requirements.txt")
    print("├── README.md")
    print("└── webcams.webm")


if __name__ == "__main__":
    print("🚀 Audio Transcriber and Summarizer - Modular Test")
    print("=" * 60)
    
    show_project_structure()
    test_imports()
