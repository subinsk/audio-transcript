"""
Llama-based summarization module.
"""

import warnings
from typing import Optional, List
import torch

# Import conditionally to avoid errors if not installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Install with: pip install transformers")

from src.config import LLAMA_MODEL_CANDIDATES, USE_GPU_IF_AVAILABLE, TORCH_DTYPE_GPU, TORCH_DTYPE_CPU

warnings.filterwarnings("ignore")


class LlamaSummarizer:
    """Handle text summarization using Llama models."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Llama summarizer.
        
        Args:
            model_path: Path to Llama model, if None will try candidates
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            warnings.warn("Transformers not available. Will use basic summarization.")
            return
        
        self._load_model()
    
    def _load_model(self):
        """Load the Llama model for summarization."""
        try:
            print("📥 Loading Llama model...")
            
            # Determine model candidates
            candidates = [self.model_path] if self.model_path else []
            candidates.extend(LLAMA_MODEL_CANDIDATES)
            
            model_loaded = False
            for candidate in candidates:
                if candidate is None:
                    continue
                    
                try:
                    print(f"   Trying: {candidate}")
                    
                    # Determine torch dtype based on GPU availability
                    use_gpu = USE_GPU_IF_AVAILABLE and torch.cuda.is_available()
                    torch_dtype = getattr(torch, TORCH_DTYPE_GPU) if use_gpu else getattr(torch, TORCH_DTYPE_CPU)
                    
                    # Try loading the model
                    self.tokenizer = AutoTokenizer.from_pretrained(candidate)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        candidate,
                        torch_dtype=torch_dtype,
                        device_map="auto" if use_gpu else None,
                        trust_remote_code=True
                    )
                    
                    # Set pad token if not set
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Create pipeline
                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch_dtype,
                        device_map="auto" if use_gpu else None
                    )
                    
                    print(f"✅ Llama model loaded successfully: {candidate}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {candidate}: {e}")
                    continue
            
            if not model_loaded:
                print("⚠️  Could not load Llama model. Using basic summarization.")
                self.pipeline = None
                
        except Exception as e:
            print(f"❌ Error loading Llama model: {e}")
            self.pipeline = None
    
    def generate_summary(self, text: str, summary_type: str = "detailed") -> str:
        """
        Generate summary using Llama model.
        
        Args:
            text: Text to summarize
            summary_type: Type of summary ('detailed', 'brief', 'key_points')
            
        Returns:
            Generated summary
        """
        if self.pipeline is None:
            return self._generate_basic_summary(text)
        
        try:
            # Prepare prompt based on summary type
            prompt = self._create_prompt(text, summary_type)
            
            # Generate response
            print(f"🤖 Generating {summary_type} summary with Llama...")
            
            response = self.pipeline(
                prompt,
                max_new_tokens=1000,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text']
            summary = generated_text.split("Summary:")[-1].strip()
            
            if not summary or len(summary) < 50:
                print("⚠️  Generated summary seems too short, falling back to basic summarization")
                return self._generate_basic_summary(text)
            
            return summary
            
        except Exception as e:
            print(f"❌ Error generating summary with Llama: {e}")
            return self._generate_basic_summary(text)
    
    def _create_prompt(self, text: str, summary_type: str) -> str:
        """Create appropriate prompt for summarization."""
        if summary_type == "detailed":
            return f"""Please provide a comprehensive and detailed summary of the following transcript. Include all important points, key topics discussed, main arguments, and relevant details. Maintain the logical flow and context:

Transcript:
{text}

Detailed Summary:"""
        elif summary_type == "brief":
            return f"""Please provide a brief, concise summary of the following transcript, highlighting only the most important points:

Transcript:
{text}

Brief Summary:"""
        else:  # key_points
            return f"""Please extract and list the key points from the following transcript in bullet format:

Transcript:
{text}

Key Points:
•"""
    
    def _generate_basic_summary(self, text: str) -> str:
        """Generate basic summary using simple text processing."""
        print("📝 Generating basic summary...")
        
        # Split into sentences
        sentences = text.split('. ')
        
        # Basic summary: take first few and last few sentences
        summary_sentences = []
        
        if len(sentences) > 10:
            summary_sentences.extend(sentences[:3])  # First 3 sentences
            summary_sentences.append("...")
            summary_sentences.extend(sentences[-3:])  # Last 3 sentences
        else:
            summary_sentences = sentences
        
        basic_summary = '. '.join(summary_sentences)
        
        # Add some basic analysis
        word_count = len(text.split())
        char_count = len(text)
        
        analysis = f"""
BASIC SUMMARY:
{basic_summary}

TRANSCRIPT ANALYSIS:
- Total words: {word_count}
- Total characters: {char_count}
- Estimated reading time: {word_count // 200} minutes
"""
        
        return analysis
    
    def chunk_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """
        Split text into chunks for processing.
        
        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk
            
        Returns:
            List of text chunks
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
