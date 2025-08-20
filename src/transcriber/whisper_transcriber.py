"""
Whisper-based transcription module.
"""

import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Import conditionally to avoid errors if not installed
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("OpenAI Whisper not available. Install with: pip install openai-whisper")

from src.config import DEFAULT_WHISPER_MODEL

warnings.filterwarnings("ignore")


class WhisperTranscriber:
    """Handle speech-to-text transcription using OpenAI Whisper."""
    
    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_name = model_name
        self.model = None
        
        if not WHISPER_AVAILABLE:
            raise ImportError("OpenAI Whisper is required. Install with: pip install openai-whisper")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model."""
        try:
            print(f"📥 Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            raise
    
    def transcribe_audio_batch(self, audio_path: str, batch_index: int = 0) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio batch using Whisper.
        
        Args:
            audio_path: Path to the audio file
            batch_index: Index of the batch for progress tracking
            
        Returns:
            Transcription result with text, segments, and metadata
        """
        if self.model is None:
            print("❌ Whisper model not loaded")
            return None
            
        try:
            print(f"🎤 Transcribing batch {batch_index}: {Path(audio_path).name}")
            
            # Transcribe using Whisper
            result = self.model.transcribe(audio_path)
            
            # Add batch metadata
            result['batch_index'] = batch_index
            result['audio_file'] = audio_path
            result['batch_filename'] = Path(audio_path).name
            
            print(f"✅ Batch {batch_index} transcribed! Language: {result.get('language', 'unknown')}, Length: {len(result['text'])} chars")
            
            return result
            
        except Exception as e:
            print(f"❌ Error transcribing batch {batch_index}: {e}")
            return None
    
    def transcribe_single_file(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Transcribe a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription result
        """
        if self.model is None:
            print("❌ Whisper model not loaded")
            return None
            
        try:
            print(f"🎤 Transcribing audio: {audio_path}")
            result = self.model.transcribe(audio_path)
            print("✅ Transcription completed!")
            print(f"📝 Detected language: {result.get('language', 'unknown')}")
            print(f"📊 Text length: {len(result['text'])} characters")
            return result
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None
