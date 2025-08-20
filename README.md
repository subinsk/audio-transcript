# Audio Transcriber and Summarizer (Batch Processing)

This project transcribes audio from video files using OpenAI Whisper and generates detailed summaries using Llama 3 8B model. It supports **batch processing** to handle long videos efficiently by splitting them into manageable chunks.

## 🌟 Features

- **Batch Audio Processing**: Automatically splits long videos into configurable time-based batches
- **High-Quality Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Intelligent Summarization**: Leverages Llama 3 8B model for generating detailed, contextual summaries
- **Multiple Summary Types**: Generates detailed, brief, and key-points summaries for each batch and final combined results
- **Organized Output Structure**: Creates a well-organized folder hierarchy for all outputs
- **Progress Tracking**: Real-time progress updates for batch processing
- **Auto-cleanup**: Removes temporary files after processing

## 📁 Output Folder Structure

```
video_name_YYYYMMDD_HHMMSS/
├── 01_audio_batches/          # Temporary audio batch files (auto-cleaned)
├── 02_transcripts_raw/        # Combined full transcript
├── 03_transcripts_batches/    # Individual batch transcripts with timestamps
├── 04_summaries/              # Final combined summaries
├── 05_summaries_batches/      # Individual batch summaries
└── 06_final_outputs/          # Master results and processing reports
```

## 🚀 Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have Llama 3 8B model available on your system. The script will try to load it from:
   - Local model path (if specified)
   - HuggingFace model hub
   - Fallback to basic summarization if model not available

## 📖 Usage

### Simple Usage (Default - 5 minute batches)
```bash
python audio_transcriber_summarizer.py
```
This will process the `webcams.webm` file in the current directory using 5-minute batches.

### Custom Video File with Default Batching
```bash
python audio_transcriber_summarizer.py path/to/your/video.webm
```

### Custom Batch Duration
```bash
python audio_transcriber_summarizer.py video.webm --batch-duration 180
```
This processes the video in 3-minute (180-second) batches.

### Advanced Usage
```bash
python audio_transcriber_summarizer.py video.webm --batch-duration 600 --whisper-model medium --summary-types detailed brief
```

### Using Batch Script (Windows)
```batch
run_transcriber.bat
run_transcriber.bat --batch-duration 240
run_transcriber.bat custom_video.mp4 --whisper-model large
```

## ⚙️ Parameters

- `--batch-duration`: Duration in seconds for each batch (default: 300 = 5 minutes)
- `--whisper-model`: Whisper model size (tiny, base, small, medium, large)
- `--llama-model`: Path to Llama 3 8B model
- `--output-dir`: Output directory for results
- `--summary-types`: Types of summaries to generate (detailed, brief, key_points)

## 📋 Output Files

### Per Batch:
- `batch_XXX_transcript.txt`: Individual batch transcript with timestamps
- `batch_XXX_detailed_summary.txt`: Detailed summary of the batch
- `batch_XXX_brief_summary.txt`: Brief summary of the batch
- `batch_XXX_key_points_summary.txt`: Key points from the batch

### Combined Results:
- `*_full_transcript_*.txt`: Complete transcript from all batches
- `*_detailed_final_summary_*.txt`: Comprehensive summary of entire video
- `*_brief_final_summary_*.txt`: Concise summary of entire video
- `*_key_points_final_summary_*.txt`: Key points from entire video
- `*_master_results_*.json`: Complete results in JSON format
- `*_processing_report_*.txt`: Processing statistics and folder structure

## 🔧 Batch Processing Benefits

1. **Memory Efficiency**: Processes large videos without overwhelming system memory
2. **Progress Tracking**: See real-time progress as each batch is completed
3. **Fault Tolerance**: If one batch fails, others continue processing
4. **Organized Results**: Easy to review individual sections or the complete result
5. **Flexibility**: Adjust batch size based on content complexity or system resources

## 💡 Recommended Batch Sizes

- **Short conversations/meetings**: 3-5 minutes (180-300 seconds)
- **Lectures/presentations**: 5-10 minutes (300-600 seconds)
- **Long documentaries**: 10-15 minutes (600-900 seconds)
- **System limitations**: Reduce batch size if running low on memory

## 🖥️ System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for Llama model)
- Sufficient RAM for model loading (8GB+ recommended)
- ffmpeg (for audio processing)
- Storage space for batch processing (temporary files)

## 📊 Model Information

### Whisper Models
- **tiny**: Fastest, lowest quality (~39 MB)
- **base**: Good balance (~74 MB) - **Recommended for batch processing**
- **small**: Better accuracy (~244 MB)
- **medium**: High accuracy (~769 MB)
- **large**: Best accuracy (~1550 MB)

### Llama 3 8B
- Requires approximately 16GB VRAM for optimal performance
- Falls back to CPU if GPU not available
- Uses quantized models when possible to reduce memory usage

## 🎯 Best Practices

1. **Start with smaller batches** (3-5 minutes) for complex audio
2. **Use base or small Whisper models** for faster batch processing
3. **Monitor disk space** during processing of very long videos
4. **Check the processing report** after completion for statistics
5. **Review individual batch results** if quality issues are suspected
