"""
Audio Transcriber and Summarizer using Whisper and Llama 3 8B
This script extracts audio from video files, transcribes it using OpenAI Whisper,
and generates detailed summaries using Llama 3 8B model.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import torch
from moviepy.editor import VideoFileClip
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

class AudioTranscriberSummarizer:
    def __init__(self, whisper_model="base", llama_model_path=None, batch_duration=300):
        """
        Initialize the transcriber and summarizer.
        
        Args:
            whisper_model (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            llama_model_path (str): Path to Llama 3 8B model, if None will try to load from HuggingFace
            batch_duration (int): Duration in seconds for each batch (default: 5 minutes)
        """
        self.whisper_model_name = whisper_model
        self.whisper_model = None
        self.llama_tokenizer = None
        self.llama_model = None
        self.llama_pipeline = None
        self.batch_duration = batch_duration
        
        print("🚀 Initializing Audio Transcriber and Summarizer...")
        self._load_whisper_model()
        self._load_llama_model(llama_model_path)
    
    def _load_whisper_model(self):
        """Load the Whisper model for transcription."""
        try:
            print(f"📥 Loading Whisper model: {self.whisper_model_name}")
            self.whisper_model = whisper.load_model(self.whisper_model_name)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading Whisper model: {e}")
            sys.exit(1)
    
    def _load_llama_model(self, model_path=None):
        """Load the Llama 3 8B model for summarization."""
        try:
            print("📥 Loading Llama 3 8B model...")
            
            # Try different possible model names/paths for Llama 3 8B
            model_candidates = [
                model_path,
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Llama-2-7b-chat-hf",
                "microsoft/DialoGPT-large"
            ]
            
            model_loaded = False
            for candidate in model_candidates:
                if candidate is None:
                    continue
                    
                try:
                    print(f"   Trying: {candidate}")
                    
                    # Check if it's a local path
                    if os.path.exists(str(candidate)):
                        self.llama_tokenizer = AutoTokenizer.from_pretrained(candidate, local_files_only=True)
                        self.llama_model = AutoModelForCausalLM.from_pretrained(
                            candidate,
                            local_files_only=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                    else:
                        # Try loading from HuggingFace
                        self.llama_tokenizer = AutoTokenizer.from_pretrained(candidate)
                        self.llama_model = AutoModelForCausalLM.from_pretrained(
                            candidate,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None,
                            trust_remote_code=True
                        )
                    
                    # Set pad token if not set
                    if self.llama_tokenizer.pad_token is None:
                        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
                    
                    # Create pipeline
                    self.llama_pipeline = pipeline(
                        "text-generation",
                        model=self.llama_model,
                        tokenizer=self.llama_tokenizer,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None
                    )
                    
                    print(f"✅ Llama model loaded successfully: {candidate}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {candidate}: {e}")
                    continue
            
            if not model_loaded:
                print("⚠️  Could not load Llama model. Using a fallback summarization approach.")
                self.llama_pipeline = None
                
        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            print("⚠️  Continuing with basic summarization...")
            self.llama_pipeline = None
    
    def extract_audio_batches(self, video_path, output_dir, batch_duration=None):
        """
        Extract audio from video file and split into batches.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Output directory for audio batches
            batch_duration (int): Duration in seconds for each batch
            
        Returns:
            list: List of audio batch file paths
        """
        if batch_duration is None:
            batch_duration = self.batch_duration
            
        try:
            video_path = Path(video_path)
            output_dir = Path(output_dir)
            
            print(f"🎵 Extracting audio from: {video_path}")
            print(f"📊 Batch duration: {batch_duration} seconds ({batch_duration//60}:{batch_duration%60:02d})")
            
            # Load video and get total duration
            video = VideoFileClip(str(video_path))
            total_duration = video.duration
            audio = video.audio
            
            print(f"� Total video duration: {total_duration:.1f} seconds ({int(total_duration//60)}:{int(total_duration%60):02d})")
            
            # Calculate number of batches
            num_batches = int(total_duration / batch_duration) + (1 if total_duration % batch_duration > 0 else 0)
            print(f"📦 Creating {num_batches} audio batches")
            
            batch_files = []
            
            for i in range(num_batches):
                start_time = i * batch_duration
                end_time = min((i + 1) * batch_duration, total_duration)
                
                # Extract audio segment
                audio_segment = audio.subclip(start_time, end_time)
                
                # Create batch file name
                batch_filename = f"batch_{i+1:03d}_{int(start_time):04d}s_{int(end_time):04d}s.wav"
                batch_path = output_dir / batch_filename
                
                # Write audio batch
                print(f"   📝 Creating batch {i+1}/{num_batches}: {batch_filename} ({end_time-start_time:.1f}s)")
                audio_segment.write_audiofile(str(batch_path), verbose=False, logger=None)
                
                batch_files.append(str(batch_path))
                
                # Clean up segment
                audio_segment.close()
            
            # Clean up
            audio.close()
            video.close()
            
            print(f"✅ Created {len(batch_files)} audio batches")
            return batch_files
            
        except Exception as e:
            print(f"❌ Error extracting audio batches: {e}")
            return []
    
    def transcribe_audio_batch(self, audio_path, batch_index=0):
        """
        Transcribe a single audio batch using Whisper.
        
        Args:
            audio_path (str): Path to the audio file
            batch_index (int): Index of the batch for progress tracking
            
        Returns:
            dict: Transcription result with text, segments, and metadata
        """
        try:
            print(f"🎤 Transcribing batch {batch_index}: {Path(audio_path).name}")
            
            # Transcribe using Whisper
            result = self.whisper_model.transcribe(audio_path)
            
            # Add batch metadata
            result['batch_index'] = batch_index
            result['audio_file'] = audio_path
            result['batch_filename'] = Path(audio_path).name
            
            print(f"✅ Batch {batch_index} transcribed! Language: {result.get('language', 'unknown')}, Length: {len(result['text'])} chars")
            
            return result
            
        except Exception as e:
            print(f"❌ Error transcribing batch {batch_index}: {e}")
            return None
    
    def chunk_text(self, text, max_chunk_size=2000):
        """
        Split text into chunks for processing.
        
        Args:
            text (str): Text to chunk
            max_chunk_size (int): Maximum characters per chunk
            
        Returns:
            list: List of text chunks
        """
        # Split by sentences first
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Add sentence to current chunk if it fits
            if len(current_chunk + sentence + '. ') <= max_chunk_size:
                current_chunk += sentence + '. '
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_summary_with_llama(self, text, summary_type="detailed"):
        """
        Generate summary using Llama 3 8B model.
        
        Args:
            text (str): Text to summarize
            summary_type (str): Type of summary ('detailed', 'brief', 'key_points')
            
        Returns:
            str: Generated summary
        """
        if self.llama_pipeline is None:
            return self.generate_basic_summary(text)
        
        try:
            # Prepare prompt based on summary type
            if summary_type == "detailed":
                prompt = f"""Please provide a comprehensive and detailed summary of the following transcript. Include all important points, key topics discussed, main arguments, and relevant details. Maintain the logical flow and context:

Transcript:
{text}

Detailed Summary:"""
            elif summary_type == "brief":
                prompt = f"""Please provide a brief, concise summary of the following transcript, highlighting only the most important points:

Transcript:
{text}

Brief Summary:"""
            else:  # key_points
                prompt = f"""Please extract and list the key points from the following transcript in bullet format:

Transcript:
{text}

Key Points:
•"""
            
            # Generate response
            print(f"🤖 Generating {summary_type} summary with Llama 3 8B...")
            
            response = self.llama_pipeline(
                prompt,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llama_tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            summary = generated_text.split("Summary:")[-1].strip()
            
            if not summary or len(summary) < 50:
                print("⚠️  Generated summary seems too short, falling back to basic summarization")
                return self.generate_basic_summary(text)
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary with Llama: {e}")
            return self.generate_basic_summary(text)
    
    def create_folder_structure(self, base_dir, video_name, timestamp):
        """
        Create organized folder structure for outputs.
        
        Args:
            base_dir (str): Base directory for outputs
            video_name (str): Name of the video file
            timestamp (str): Timestamp for unique folder naming
            
        Returns:
            dict: Dictionary containing all folder paths
        """
        base_dir = Path(base_dir)
        
        # Main project folder
        project_folder = base_dir / f"{video_name}_{timestamp}"
        
        # Subfolders
        folders = {
            'project': project_folder,
            'audio_batches': project_folder / '01_audio_batches',
            'transcripts_raw': project_folder / '02_transcripts_raw',
            'transcripts_batches': project_folder / '03_transcripts_batches',
            'summaries': project_folder / '04_summaries',
            'summaries_batches': project_folder / '05_summaries_batches',
            'final_outputs': project_folder / '06_final_outputs'
        }
        
        # Create all folders
        for folder_name, folder_path in folders.items():
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created folder: {folder_path}")
        
        return folders
    
    def save_batch_transcript(self, transcript_result, folders, batch_index):
        """
        Save individual batch transcript to file.
        
        Args:
            transcript_result (dict): Transcription result
            folders (dict): Folder structure dictionary
            batch_index (int): Batch index
            
        Returns:
            str: Path to saved transcript file
        """
        try:
            batch_filename = f"batch_{batch_index+1:03d}_transcript.txt"
            transcript_path = folders['transcripts_batches'] / batch_filename
            
            with open(transcript_path, 'w', encoding='utf-8') as f:
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
    
    def save_batch_summary(self, summary_text, folders, batch_index, summary_type):
        """
        Save individual batch summary to file.
        
        Args:
            summary_text (str): Summary text
            folders (dict): Folder structure dictionary
            batch_index (int): Batch index
            summary_type (str): Type of summary
            
        Returns:
            str: Path to saved summary file
        """
        try:
            summary_filename = f"batch_{batch_index+1:03d}_{summary_type}_summary.txt"
            summary_path = folders['summaries_batches'] / summary_filename
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"BATCH {batch_index+1} - {summary_type.upper()} SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(summary_text)
                f.write("\n\n" + "=" * 50)
            
            print(f"💾 Batch {batch_index+1} {summary_type} summary saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            print(f"❌ Error saving batch {batch_index+1} {summary_type} summary: {e}")
            return None
    
    def process_video(self, video_path, output_dir=None, summary_types=None, batch_duration=None):
        """
        Complete batch processing pipeline: extract audio, transcribe, and summarize in batches.
        
        Args:
            video_path (str): Path to the video file
            output_dir (str): Output directory for results
            summary_types (list): Types of summaries to generate
            batch_duration (int): Duration in seconds for each batch
            
        Returns:
            dict: Results containing all batch transcriptions and summaries
        """
        if summary_types is None:
            summary_types = ["detailed", "brief", "key_points"]
        
        if output_dir is None:
            output_dir = Path(video_path).parent
        
        if batch_duration is None:
            batch_duration = self.batch_duration
        
        video_path = Path(video_path)
        video_name = video_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized folder structure
        print(f"\n🏗️  Creating folder structure for: {video_name}")
        folders = self.create_folder_structure(output_dir, video_name, timestamp)
        
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
            batch_files = self.extract_audio_batches(video_path, folders['audio_batches'], batch_duration)
            
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
                transcript_result = self.transcribe_audio_batch(batch_file, i)
                if not transcript_result:
                    print(f"⚠️  Skipping batch {i+1} due to transcription error")
                    continue
                
                # Save batch transcript
                self.save_batch_transcript(transcript_result, folders, i)
                
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
                        batch_summary = self.generate_summary_with_llama(batch_text, summary_type)
                        
                        # Save batch summary
                        self.save_batch_summary(batch_summary, folders, i, summary_type)
                        
                        # Store summary
                        batch_summaries[summary_type].append({
                            "batch_index": i + 1,
                            "summary": batch_summary
                        })
                else:
                    print(f"   ⚠️  Batch {i+1} has insufficient content for summarization")
                
                # Clean up audio batch file
                try:
                    os.remove(batch_file)
                except:
                    pass
                
                print(f"✅ Batch {i+1} completed!")
            
            # Step 3: Combine all transcripts
            print(f"\n📝 STEP 3: Combining transcripts...")
            combined_transcript_parts = []
            for transcript_result in all_transcripts:
                batch_header = f"\n{'='*60}\nBATCH {transcript_result['batch_index']+1}\n{'='*60}\n"
                combined_transcript_parts.append(batch_header + transcript_result['text'])
            
            combined_transcript = "\n\n".join(combined_transcript_parts)
            results["combined_transcript"] = combined_transcript
            
            # Save combined transcript
            combined_transcript_file = folders['transcripts_raw'] / f"{video_name}_full_transcript_{timestamp}.txt"
            with open(combined_transcript_file, 'w', encoding='utf-8') as f:
                f.write(f"COMPLETE TRANSCRIPT - {video_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Batches: {len(all_transcripts)}\n")
                f.write(f"Batch Duration: {batch_duration} seconds\n")
                f.write("="*80 + "\n\n")
                f.write(combined_transcript)
            
            print(f"💾 Combined transcript saved: {combined_transcript_file}")
            
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
                    final_summary = self.generate_summary_with_llama(combined_summaries, summary_type)
                else:
                    # Use full transcript for final summary
                    final_summary = self.generate_summary_with_llama(combined_transcript, summary_type)
                
                results["final_summaries"][summary_type] = final_summary
                
                # Save final summary
                final_summary_file = folders['summaries'] / f"{video_name}_{summary_type}_final_summary_{timestamp}.txt"
                with open(final_summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"FINAL {summary_type.upper()} SUMMARY - {video_name}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total Batches Processed: {len(all_transcripts)}\n")
                    f.write("="*80 + "\n\n")
                    f.write(final_summary)
                
                print(f"💾 Final {summary_type} summary saved: {final_summary_file}")
            
            # Store batch summaries in results
            results["batch_summaries"] = batch_summaries
            
            # Step 5: Create master results file
            print(f"\n📊 STEP 5: Creating master results file...")
            
            master_results_file = folders['final_outputs'] / f"{video_name}_master_results_{timestamp}.json"
            with open(master_results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Create summary report
            summary_report_file = folders['final_outputs'] / f"{video_name}_processing_report_{timestamp}.txt"
            with open(summary_report_file, 'w', encoding='utf-8') as f:
                f.write(f"PROCESSING REPORT - {video_name}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video File: {video_path}\n")
                f.write(f"Batch Duration: {batch_duration} seconds\n")
                f.write(f"Total Batches: {len(all_transcripts)}\n")
                f.write(f"Total Transcript Length: {len(combined_transcript)} characters\n")
                f.write(f"Summary Types: {', '.join(summary_types)}\n\n")
                
                f.write("FOLDER STRUCTURE:\n")
                for folder_name, folder_path in folders.items():
                    f.write(f"  {folder_name}: {folder_path}\n")
                
                f.write(f"\nBATCH PROCESSING RESULTS:\n")
                for i, transcript_data in enumerate(results["batch_transcriptions"]):
                    f.write(f"  Batch {i+1}: {transcript_data['segments']} segments, {len(transcript_data['transcript'])} chars\n")
            
            print(f"💾 Processing report saved: {summary_report_file}")
            print(f"💾 Master results saved: {master_results_file}")
            
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


def main():
    parser = argparse.ArgumentParser(description="Transcribe and summarize video files in batches")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--whisper-model", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--llama-model", default=None,
                       help="Path to Llama 3 8B model")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory for results")
    parser.add_argument("--summary-types", nargs="+", 
                       default=["detailed", "brief", "key_points"],
                       choices=["detailed", "brief", "key_points"],
                       help="Types of summaries to generate")
    parser.add_argument("--batch-duration", type=int, default=300,
                       help="Duration in seconds for each batch (default: 300 = 5 minutes)")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"❌ Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Initialize transcriber and summarizer
    transcriber = AudioTranscriberSummarizer(
        whisper_model=args.whisper_model,
        llama_model_path=args.llama_model,
        batch_duration=args.batch_duration
    )
    
    # Process the video
    print(f"\n🎬 Processing video: {args.video_path}")
    print(f"⏱️  Batch duration: {args.batch_duration} seconds ({args.batch_duration//60}:{args.batch_duration%60:02d})")
    
    results = transcriber.process_video(
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


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, use the webcams.webm file in current directory
        video_file = "webcams.webm"
        if os.path.exists(video_file):
            print(f"🎬 Processing default video file: {video_file}")
            print("⏱️  Using default batch duration: 300 seconds (5 minutes)")
            
            transcriber = AudioTranscriberSummarizer(batch_duration=300)
            results = transcriber.process_video(video_file)
            
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
        else:
            print(f"❌ Default video file '{video_file}' not found!")
            print("Usage: python audio_transcriber_summarizer.py <video_path>")
            print("Optional arguments:")
            print("  --batch-duration SECONDS    Duration for each batch (default: 300)")
            print("  --whisper-model MODEL        Whisper model size (default: base)")
            print("  --summary-types TYPE [TYPE]  Summary types to generate")
    else:
        main()
