# Configuration file for Audio Transcriber and Summarizer

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
