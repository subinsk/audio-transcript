"""
File management utilities for organizing outputs.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.config import FOLDER_STRUCTURE, TIMESTAMP_FORMAT, DATETIME_FORMAT, DEFAULT_ENCODING


class FileManager:
    """Handle file operations and folder structure management."""
    
    def __init__(self):
        """Initialize the file manager."""
        pass
    
    def create_folder_structure(self, base_dir: str, video_name: str, timestamp: str) -> Dict[str, Path]:
        """
        Create organized folder structure for outputs.
        
        Args:
            base_dir: Base directory for outputs
            video_name: Name of the video file
            timestamp: Timestamp for unique folder naming
            
        Returns:
            Dictionary containing all folder paths
        """
        base_dir = Path(base_dir)
        
        # Main project folder
        project_folder = base_dir / f"{video_name}_{timestamp}"
        
        # Subfolders
        folders = {
            'project': project_folder
        }
        
        for key, folder_name in FOLDER_STRUCTURE.items():
            folders[key] = project_folder / folder_name
        
        # Create all folders
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created folder: {folder_path}")
        
        return folders
    
    def save_batch_transcript(self, transcript_result: Dict[str, Any], 
                            folders: Dict[str, Path], batch_index: int) -> str:
        """
        Save individual batch transcript to file.
        
        Args:
            transcript_result: Transcription result
            folders: Folder structure dictionary
            batch_index: Batch index
            
        Returns:
            Path to saved transcript file
        """
        try:
            batch_filename = f"batch_{batch_index+1:03d}_transcript.txt"
            transcript_path = folders['transcripts_batches'] / batch_filename
            
            with open(transcript_path, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(f"BATCH {batch_index+1} TRANSCRIPT\n")
                f.write(f"Audio File: {transcript_result.get('batch_filename', 'Unknown')}\n")
                f.write(f"Language: {transcript_result.get('language', 'Unknown')}\n")
                f.write(f"Duration: {len(transcript_result.get('segments', []))} segments\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcript_result['text'])
                f.write("\n\n" + "=" * 50 + "\n")
                f.write("SEGMENTS WITH TIMESTAMPS:\n\n")
                
                for segment in transcript_result.get('segments', []):
                    start_time = segment.get('start', 0)
                    end_time = segment.get('end', 0)
                    text = segment.get('text', '')
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {text.strip()}\n")
            
            print(f"💾 Batch {batch_index+1} transcript saved: {transcript_path}")
            return str(transcript_path)
            
        except Exception as e:
            print(f"❌ Error saving batch {batch_index+1} transcript: {e}")
            return None
    
    def save_batch_summary(self, summary_text: str, folders: Dict[str, Path], 
                         batch_index: int, summary_type: str) -> str:
        """
        Save individual batch summary to file.
        
        Args:
            summary_text: Summary text
            folders: Folder structure dictionary
            batch_index: Batch index
            summary_type: Type of summary
            
        Returns:
            Path to saved summary file
        """
        try:
            summary_filename = f"batch_{batch_index+1:03d}_{summary_type}_summary.txt"
            summary_path = folders['summaries_batches'] / summary_filename
            
            with open(summary_path, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(f"BATCH {batch_index+1} - {summary_type.upper()} SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary_text)
                f.write("\n\n" + "=" * 50)
            
            print(f"💾 Batch {batch_index+1} {summary_type} summary saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            print(f"❌ Error saving batch {batch_index+1} {summary_type} summary: {e}")
            return None
    
    def save_combined_transcript(self, transcripts: List[Dict[str, Any]], 
                               folders: Dict[str, Path], video_name: str, timestamp: str) -> str:
        """
        Save combined transcript from all batches.
        
        Args:
            transcripts: List of transcript results
            folders: Folder structure dictionary
            video_name: Name of the video
            timestamp: Processing timestamp
            
        Returns:
            Path to saved combined transcript
        """
        try:
            combined_transcript_parts = []
            for transcript_result in transcripts:
                batch_header = f"\n{'='*60}\nBATCH {transcript_result['batch_index']+1}\n{'='*60}\n"
                combined_transcript_parts.append(batch_header + transcript_result['text'])
            
            combined_transcript = "\n\n".join(combined_transcript_parts)
            
            # Save combined transcript
            combined_transcript_file = folders['transcripts_raw'] / f"{video_name}_full_transcript_{timestamp}.txt"
            with open(combined_transcript_file, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(f"COMPLETE TRANSCRIPT - {video_name}\n")
                f.write(f"Generated: {datetime.now().strftime(DATETIME_FORMAT)}\n")
                f.write(f"Total Batches: {len(transcripts)}\n")
                f.write("="*80 + "\n\n")
                f.write(combined_transcript)
            
            print(f"💾 Combined transcript saved: {combined_transcript_file}")
            return str(combined_transcript_file)
            
        except Exception as e:
            print(f"❌ Error saving combined transcript: {e}")
            return None
    
    def save_final_summary(self, summary_text: str, folders: Dict[str, Path], 
                         video_name: str, timestamp: str, summary_type: str, 
                         num_batches: int) -> str:
        """
        Save final summary to file.
        
        Args:
            summary_text: Summary text
            folders: Folder structure dictionary
            video_name: Name of the video
            timestamp: Processing timestamp
            summary_type: Type of summary
            num_batches: Number of batches processed
            
        Returns:
            Path to saved summary file
        """
        try:
            final_summary_file = folders['summaries'] / f"{video_name}_{summary_type}_final_summary_{timestamp}.txt"
            with open(final_summary_file, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(f"FINAL {summary_type.upper()} SUMMARY - {video_name}\n")
                f.write(f"Generated: {datetime.now().strftime(DATETIME_FORMAT)}\n")
                f.write(f"Total Batches Processed: {num_batches}\n")
                f.write("="*80 + "\n\n")
                f.write(summary_text)
            
            print(f"💾 Final {summary_type} summary saved: {final_summary_file}")
            return str(final_summary_file)
            
        except Exception as e:
            print(f"❌ Error saving final {summary_type} summary: {e}")
            return None
    
    def save_master_results(self, results: Dict[str, Any], folders: Dict[str, Path], 
                          video_name: str, timestamp: str) -> str:
        """
        Save master results file in JSON format.
        
        Args:
            results: Complete results dictionary
            folders: Folder structure dictionary
            video_name: Name of the video
            timestamp: Processing timestamp
            
        Returns:
            Path to saved results file
        """
        try:
            master_results_file = folders['final_outputs'] / f"{video_name}_master_results_{timestamp}.json"
            with open(master_results_file, 'w', encoding=DEFAULT_ENCODING) as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Master results saved: {master_results_file}")
            return str(master_results_file)
            
        except Exception as e:
            print(f"❌ Error saving master results: {e}")
            return None
    
    def save_processing_report(self, results: Dict[str, Any], folders: Dict[str, Path], 
                             video_name: str, timestamp: str, video_path: str) -> str:
        """
        Save processing report with statistics.
        
        Args:
            results: Complete results dictionary
            folders: Folder structure dictionary
            video_name: Name of the video
            timestamp: Processing timestamp
            video_path: Original video path
            
        Returns:
            Path to saved report file
        """
        try:
            summary_report_file = folders['final_outputs'] / f"{video_name}_processing_report_{timestamp}.txt"
            with open(summary_report_file, 'w', encoding=DEFAULT_ENCODING) as f:
                f.write(f"PROCESSING REPORT - {video_name}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Processing Date: {datetime.now().strftime(DATETIME_FORMAT)}\n")
                f.write(f"Video File: {video_path}\n")
                f.write(f"Batch Duration: {results.get('batch_duration', 'Unknown')} seconds\n")
                f.write(f"Total Batches: {len(results.get('batch_transcriptions', []))}\n")
                f.write(f"Total Transcript Length: {len(results.get('combined_transcript', ''))} characters\n")
                f.write(f"Summary Types: {', '.join(results.get('final_summaries', {}).keys())}\n\n")
                
                f.write("FOLDER STRUCTURE:\n")
                for folder_name, folder_path in folders.items():
                    f.write(f"  {folder_name}: {folder_path}\n")
                
                f.write(f"\nBATCH PROCESSING RESULTS:\n")
                for i, transcript_data in enumerate(results.get("batch_transcriptions", [])):
                    f.write(f"  Batch {i+1}: {transcript_data.get('segments', 0)} segments, {len(transcript_data.get('transcript', ''))} chars\n")
            
            print(f"💾 Processing report saved: {summary_report_file}")
            return str(summary_report_file)
            
        except Exception as e:
            print(f"❌ Error saving processing report: {e}")
            return None
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary files.
        
        Args:
            file_paths: List of file paths to remove
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🧹 Cleaned up: {Path(file_path).name}")
            except Exception as e:
                print(f"⚠️  Could not clean up {file_path}: {e}")
    
    @staticmethod
    def generate_timestamp() -> str:
        """Generate timestamp string for unique naming."""
        return datetime.now().strftime(TIMESTAMP_FORMAT)
