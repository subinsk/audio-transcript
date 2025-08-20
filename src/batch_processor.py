"""
Main batch processor that orchestrates the entire pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.processors.audio_processor import AudioProcessor
from src.transcriber.whisper_transcriber import WhisperTranscriber
from src.summarizer.llama_summarizer import LlamaSummarizer
from src.utils.file_manager import FileManager
from src.config import (
    DEFAULT_SUMMARY_TYPES, 
    BATCH_DURATION_SECONDS,
    AUTO_CLEANUP_TEMP_FILES
)


class BatchProcessor:
    """Main class that orchestrates the entire batch processing pipeline."""
    
    def __init__(self, whisper_model: str = "base", llama_model_path: Optional[str] = None,
                 batch_duration: int = BATCH_DURATION_SECONDS):
        """
        Initialize the batch processor.
        
        Args:
            whisper_model: Whisper model size
            llama_model_path: Path to Llama model
            batch_duration: Duration in seconds for each batch
        """
        self.batch_duration = batch_duration
        
        # Initialize components
        print("🚀 Initializing Batch Processor...")
        self.audio_processor = AudioProcessor(batch_duration)
        self.transcriber = WhisperTranscriber(whisper_model)
        self.summarizer = LlamaSummarizer(llama_model_path)
        self.file_manager = FileManager()
        
        print("✅ Batch Processor initialized successfully!")
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None,
                     summary_types: Optional[List[str]] = None,
                     batch_duration: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete batch processing pipeline.
        
        Args:
            video_path: Path to the video file
            output_dir: Output directory for results
            summary_types: Types of summaries to generate
            batch_duration: Duration in seconds for each batch
            
        Returns:
            Results containing all batch transcriptions and summaries
        """
        if summary_types is None:
            summary_types = DEFAULT_SUMMARY_TYPES.copy()
        
        if output_dir is None:
            output_dir = Path(video_path).parent
        
        if batch_duration is None:
            batch_duration = self.batch_duration
        
        video_path = Path(video_path)
        video_name = video_path.stem
        timestamp = self.file_manager.generate_timestamp()
        
        # Create organized folder structure
        print(f"\n🏗️  Creating folder structure for: {video_name}")
        folders = self.file_manager.create_folder_structure(output_dir, video_name, timestamp)
        
        results = {
            "video_file": str(video_path),
            "timestamp": timestamp,
            "batch_duration": batch_duration,
            "folders": {k: str(v) for k, v in folders.items()},
            "batch_transcriptions": [],
            "batch_summaries": {},
            "combined_transcript": "",
            "final_summaries": {}
        }
        
        try:
            # Step 1: Extract audio in batches
            print(f"\n📦 STEP 1: Extracting audio batches...")
            batch_files = self.audio_processor.extract_audio_batches(
                video_path, folders['audio_batches'], batch_duration
            )
            
            if not batch_files:
                print("❌ Failed to extract audio batches")
                return results
            
            # Step 2: Process each batch
            print(f"\n🎤 STEP 2: Processing {len(batch_files)} batches...")
            
            all_transcripts = []
            batch_summaries = {summary_type: [] for summary_type in summary_types}
            
            for i, batch_file in enumerate(batch_files):
                print(f"\n📋 Processing Batch {i+1}/{len(batch_files)}")
                print("-" * 50)
                
                # Transcribe batch
                transcript_result = self.transcriber.transcribe_audio_batch(batch_file, i)
                if not transcript_result:
                    print(f"⚠️  Skipping batch {i+1} due to transcription error")
                    continue
                
                # Save batch transcript
                self.file_manager.save_batch_transcript(transcript_result, folders, i)
                
                # Store transcript
                all_transcripts.append(transcript_result)
                results["batch_transcriptions"].append({
                    "batch_index": i + 1,
                    "audio_file": batch_file,
                    "transcript": transcript_result['text'],
                    "language": transcript_result.get('language', 'unknown'),
                    "segments": len(transcript_result.get('segments', []))
                })
                
                # Generate summaries for this batch
                batch_text = transcript_result['text']
                if len(batch_text.strip()) > 10:  # Only summarize if there's meaningful content
                    for summary_type in summary_types:
                        print(f"   🤖 Generating {summary_type} summary for batch {i+1}...")
                        batch_summary = self.summarizer.generate_summary(batch_text, summary_type)
                        
                        # Save batch summary
                        self.file_manager.save_batch_summary(batch_summary, folders, i, summary_type)
                        
                        # Store summary
                        batch_summaries[summary_type].append({
                            "batch_index": i + 1,
                            "summary": batch_summary
                        })
                else:
                    print(f"   ⚠️  Batch {i+1} has insufficient content for summarization")
                
                print(f"✅ Batch {i+1} completed!")
            
            # Cleanup audio batch files
            if AUTO_CLEANUP_TEMP_FILES:
                self.file_manager.cleanup_temp_files(batch_files)
            
            # Step 3: Combine all transcripts
            print(f"\n📝 STEP 3: Combining transcripts...")
            combined_transcript_file = self.file_manager.save_combined_transcript(
                all_transcripts, folders, video_name, timestamp
            )
            
            # Get combined transcript text
            combined_transcript_parts = []
            for transcript_result in all_transcripts:
                batch_header = f"\n{'='*60}\nBATCH {transcript_result['batch_index']+1}\n{'='*60}\n"
                combined_transcript_parts.append(batch_header + transcript_result['text'])
            
            combined_transcript = "\n\n".join(combined_transcript_parts)
            results["combined_transcript"] = combined_transcript
            
            # Step 4: Generate final summaries from all content
            print(f"\n🤖 STEP 4: Generating final summaries...")
            
            for summary_type in summary_types:
                print(f"   📋 Creating final {summary_type} summary...")
                
                # Combine all batch summaries for this type
                batch_summaries_text = []
                for batch_summary in batch_summaries[summary_type]:
                    batch_summaries_text.append(f"Batch {batch_summary['batch_index']}: {batch_summary['summary']}")
                
                combined_summaries = "\n\n".join(batch_summaries_text)
                
                # Generate final summary from combined content
                if len(combined_transcript) > 4000:
                    # Use batch summaries as input for final summary
                    final_summary = self.summarizer.generate_summary(combined_summaries, summary_type)
                else:
                    # Use full transcript for final summary
                    final_summary = self.summarizer.generate_summary(combined_transcript, summary_type)
                
                results["final_summaries"][summary_type] = final_summary
                
                # Save final summary
                self.file_manager.save_final_summary(
                    final_summary, folders, video_name, timestamp, summary_type, len(all_transcripts)
                )
            
            # Store batch summaries in results
            results["batch_summaries"] = batch_summaries
            
            # Step 5: Create master results and report files
            print(f"\n📊 STEP 5: Creating master results file...")
            
            self.file_manager.save_master_results(results, folders, video_name, timestamp)
            self.file_manager.save_processing_report(
                results, folders, video_name, timestamp, str(video_path)
            )
            
            print(f"\n✅ COMPLETE! All processing finished successfully!")
            print(f"📁 Output folder: {folders['project']}")
            print(f"📊 Processed {len(all_transcripts)} batches")
            print(f"📝 Generated {len(summary_types)} types of summaries")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in batch processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        return self.audio_processor.get_video_info(video_path)
