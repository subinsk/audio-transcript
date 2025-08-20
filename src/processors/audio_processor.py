"""
Audio processing utilities for video files.
"""

import os
import warnings
from pathlib import Path
from typing import List, Optional

# Import conditionally to avoid errors if not installed
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    warnings.warn("MoviePy not available. Video processing will be limited.")

from src.config import BATCH_DURATION_SECONDS

warnings.filterwarnings("ignore")


class AudioProcessor:
    """Handle audio extraction and batch processing from video files."""
    
    def __init__(self, batch_duration: int = BATCH_DURATION_SECONDS):
        """
        Initialize the audio processor.
        
        Args:
            batch_duration: Duration in seconds for each batch
        """
        self.batch_duration = batch_duration
        
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy is required for audio processing. Install with: pip install moviepy")
    
    def extract_audio_batches(self, video_path: str, output_dir: str, 
                            batch_duration: Optional[int] = None) -> List[str]:
        """
        Extract audio from video file and split into batches.
        
        Args:
            video_path: Path to the video file
            output_dir: Output directory for audio batches
            batch_duration: Duration in seconds for each batch
            
        Returns:
            List of audio batch file paths
        """
        if batch_duration is None:
            batch_duration = self.batch_duration
            
        try:
            video_path = Path(video_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"🎵 Extracting audio from: {video_path}")
            print(f"📊 Batch duration: {batch_duration} seconds ({batch_duration//60}:{batch_duration%60:02d})")
            
            # Load video and get total duration
            video = VideoFileClip(str(video_path))
            total_duration = video.duration
            audio = video.audio
            
            print(f"🎬 Total video duration: {total_duration:.1f} seconds ({int(total_duration//60)}:{int(total_duration%60):02d})")
            
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
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video information
        """
        try:
            video = VideoFileClip(str(video_path))
            info = {
                "duration": video.duration,
                "fps": video.fps,
                "size": video.size,
                "has_audio": video.audio is not None
            }
            video.close()
            return info
        except Exception as e:
            print(f"❌ Error getting video info: {e}")
            return {}
