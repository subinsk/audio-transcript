"""
Configuration settings for the Audio Transcriber and Summarizer.
"""

import os
from pathlib import Path

# Batch Processing Settings
BATCH_DURATION_SECONDS = 300  # 5 minutes default
MIN_BATCH_DURATION = 60       # Minimum 1 minute
MAX_BATCH_DURATION = 1800     # Maximum 30 minutes

# Whisper Model Settings
DEFAULT_WHISPER_MODEL = "base"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]

# Summary Settings
DEFAULT_SUMMARY_TYPES = ["detailed", "brief", "key_points"]

# File Processing Settings
MAX_CHUNK_SIZE = 2000         # Characters per chunk for long text processing
MIN_TEXT_LENGTH = 10          # Minimum text length to attempt summarization

# Folder Names
FOLDER_STRUCTURE = {
    "audio_batches": "01_audio_batches",
    "transcripts_raw": "02_transcripts_raw", 
    "transcripts_batches": "03_transcripts_batches",
    "summaries": "04_summaries",
    "summaries_batches": "05_summaries_batches",
    "final_outputs": "06_final_outputs"
}

# Model Performance Settings
FALLBACK_TO_BASIC_SUMMARY = True  # Use basic summary if Llama model fails
AUTO_CLEANUP_TEMP_FILES = True    # Remove temporary audio files after processing
SAVE_DETAILED_LOGS = True         # Save processing logs and reports

# Llama Model Candidates (in order of preference)
LLAMA_MODEL_CANDIDATES = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "microsoft/DialoGPT-large"
]

# Audio/Video Processing
SUPPORTED_VIDEO_FORMATS = [".webm", ".mp4", ".avi", ".mov", ".mkv"]
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".aac"]

# GPU/CPU Settings
USE_GPU_IF_AVAILABLE = True
TORCH_DTYPE_GPU = "float16"
TORCH_DTYPE_CPU = "float32"

# Output Settings
DEFAULT_ENCODING = "utf-8"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
