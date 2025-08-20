#!/usr/bin/env python3
"""
Modular Audio Transcriber and Summarizer
Main entry point for batch processing video files.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.batch_processor import BatchProcessor
from src.config import (
    DEFAULT_WHISPER_MODEL, 
    DEFAULT_SUMMARY_TYPES, 
    BATCH_DURATION_SECONDS,
    WHISPER_MODELS
)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize video files in batches using modular architecture"
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, 
                       choices=WHISPER_MODELS,
                       help=f"Whisper model size (default: {DEFAULT_WHISPER_MODEL})")
    parser.add_argument("--llama-model", default=None,
                       help="Path to Llama 3 8B model")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results")
    parser.add_argument("--summary-types", nargs="+", 
                       default=DEFAULT_SUMMARY_TYPES,
                       choices=["detailed", "brief", "key_points"],
                       help="Types of summaries to generate")
    parser.add_argument("--batch-duration", type=int, default=BATCH_DURATION_SECONDS,
                       help=f"Duration in seconds for each batch (default: {BATCH_DURATION_SECONDS})")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        # Initialize batch processor
        print(f"🎬 Initializing batch processor...")
        processor = BatchProcessor(
            whisper_model=args.whisper_model,
            llama_model_path=args.llama_model,
            batch_duration=args.batch_duration
        )
        
        # Process the video
        print(f"\n🎬 Processing video: {args.video_path}")
        print(f"⏱️  Batch duration: {args.batch_duration} seconds ({args.batch_duration//60}:{args.batch_duration%60:02d})")
        
        results = processor.process_video(
            args.video_path,
            args.output_dir,
            args.summary_types,
            args.batch_duration
        )
        
        if results["batch_transcriptions"]:
            print("\n✅ Batch processing completed successfully!")
            print(f"📝 Total batches processed: {len(results['batch_transcriptions'])}")
            print(f"📊 Summaries generated: {list(results['final_summaries'].keys())}")
            print(f"📁 Output folder: {results['folders']['project']}")
        else:
            print("\n❌ Batch processing failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def process_default_video():
    """Process the default webcams.webm file if it exists."""
    video_file = "webcams.webm"
    if os.path.exists(video_file):
        print(f"🎬 Processing default video file: {video_file}")
        print("⏱️  Using default batch duration: 300 seconds (5 minutes)")
        
        try:
            processor = BatchProcessor(batch_duration=300)
            results = processor.process_video(video_file)
            
            if results["batch_transcriptions"]:
                print("\n✅ Batch processing completed successfully!")
                print(f"📝 Total batches processed: {len(results['batch_transcriptions'])}")
                print(f"📁 Output folder: {results['folders']['project']}")
                
                # Show preview of results
                print(f"\n📋 PROCESSING SUMMARY:")
                for i, batch in enumerate(results["batch_transcriptions"]):
                    print(f"  Batch {i+1}: {len(batch['transcript'])} chars, {batch['segments']} segments")
                
                print(f"\n📊 FINAL SUMMARIES GENERATED:")
                for summary_type in results['final_summaries']:
                    summary_length = len(results['final_summaries'][summary_type])
                    print(f"  {summary_type.title()}: {summary_length} characters")
            else:
                print("\n❌ Batch processing failed!")
                
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Default video file '{video_file}' not found!")
        print("Usage: python main.py <video_path>")
        print("Optional arguments:")
        print("  --batch-duration SECONDS    Duration for each batch (default: 300)")
        print("  --whisper-model MODEL        Whisper model size (default: base)")
        print("  --summary-types TYPE [TYPE]  Summary types to generate")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, use the webcams.webm file in current directory
        process_default_video()
    else:
        main()
