import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pypdf import PdfReader
from datetime import datetime
import numpy as np
import onnxruntime as ort
from transformers import WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor
import librosa
import warnings
warnings.filterwarnings("ignore")


class Document:
    """
    Simple Document class to maintain compatibility with the existing code.
    Replaces langchain_core.documents.Document.
    """
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


def load_pdfs_from_directory(directory_path: str) -> List[List[Document]]:
    """
    Search for all PDFs in a directory and extract their content with metadata.
    Returns a list where each element is a list of Document objects for one PDF.
    
    Args:
        directory_path (str): Path to the directory containing PDFs
        
    Returns:
        List[List[Document]]: List of lists, where each inner list contains 
                             Document objects (pages) for one PDF
    """
    directory = Path(directory_path)
    
    # Validate directory
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")
    
    # Find all PDF files
    pdf_files = []
    for file_path in directory.rglob("*.pdf"):
        if file_path.is_file():
            pdf_files.append(file_path)
    
    if not pdf_files:
        print(f"No PDF files found in directory: {directory}")
        return []
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    all_pdf_documents = []  # List of lists - each inner list is for one PDF
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        try:
            # Get file metadata
            file_size = pdf_path.stat().st_size
            modified_date = datetime.fromtimestamp(pdf_path.stat().st_mtime)
            
            # Load PDF content using pypdf directly
            pdf_reader = PdfReader(str(pdf_path))
            total_pages = len(pdf_reader.pages)
            pdf_documents = []  # Documents for this specific PDF
            
            # Process each page
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                
                # Extract text content from the page
                page_content = page.extract_text()
                
                # Create enhanced metadata
                enhanced_metadata = {
                    'source': str(pdf_path),
                    'pdf_name': pdf_path.name,
                    'page_number': page_num + 1,
                    'total_pages': total_pages,
                    'file_size': file_size,
                    'modified_date': modified_date.isoformat(),
                    'page': page_num  # Keep original page numbering for compatibility
                }
                
                # Create Document with enhanced metadata
                enhanced_doc = Document(
                    page_content=page_content,
                    metadata=enhanced_metadata
                )
                
                pdf_documents.append(enhanced_doc)
            
            # Add this PDF's documents to the main list
            all_pdf_documents.append(pdf_documents)
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
    
    return all_pdf_documents


class WhisperONNXTranscriber:
    """
    ONNX-based Whisper transcriber for audio files.
    Supports multiple audio formats using only ONNX models.
    """
    
    def __init__(self, model_path: str, processor_path: Optional[str] = None):
        """
        Initialize the Whisper ONNX transcriber.
        
        Args:
            model_path: Path to the directory containing ONNX models
            processor_path: Optional path to processor files (defaults to model_path)
        """
        self.model_path = Path(model_path)
        self.processor_path = Path(processor_path) if processor_path else self.model_path
        
        # Validate paths
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        if not self.processor_path.exists():
            raise ValueError(f"Processor path does not exist: {self.processor_path}")
        
        # Setup ONNX runtime providers - try multiple options
        available_providers = ort.get_available_providers()
        print(f"Available ONNX providers: {available_providers}")
        
        # Try different provider combinations
        provider_options = [
            # Option 1: DirectML (for Windows GPU acceleration)
            ['DmlExecutionProvider', 'CPUExecutionProvider'] if 'DmlExecutionProvider' in available_providers else None,
            # Option 2: OpenVINO (for ARM optimization)
            ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if 'OpenVINOExecutionProvider' in available_providers else None,
            # Option 3: CPU only
            ['CPUExecutionProvider']
        ]
        
        # Remove None options
        provider_options = [opt for opt in provider_options if opt is not None]
        
        providers = provider_options[0]  # Start with the first available option
        print(f"Using providers: {providers}")
        
        try:
            # Try to load processor from local path first, fallback to Hugging Face
            print("Loading Whisper processor...")
            try:
                self.processor = WhisperProcessor.from_pretrained(str(self.processor_path))
                print("✓ Loaded processor from local path")
            except:
                print("Local processor not found, downloading from Hugging Face...")
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                print("✓ Loaded processor from Hugging Face")
            
            print("Loading Whisper ONNX models...")
            # Load encoder and decoder ONNX models
            encoder_path = self.model_path / "encoder_model.onnx"
            decoder_path = self.model_path / "decoder_model.onnx"
            
            if not encoder_path.exists():
                raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
            if not decoder_path.exists():
                raise FileNotFoundError(f"Decoder model not found: {decoder_path}")
            
            # Try different provider combinations until one works
            self.encoder_session = None
            self.decoder_session = None
            for i, provider_set in enumerate(provider_options):
                try:
                    print(f"Trying provider option {i+1}: {provider_set}")
                    
                    # Initialize ONNX runtime sessions
                    session_options = ort.SessionOptions()
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                    
                    self.encoder_session = ort.InferenceSession(
                        str(encoder_path), 
                        sess_options=session_options,
                        providers=provider_set
                    )
                    
                    self.decoder_session = ort.InferenceSession(
                        str(decoder_path), 
                        sess_options=session_options,
                        providers=provider_set
                    )
                    
                    print(f"✓ Successfully loaded models with providers: {provider_set}")
                    break
                    
                except Exception as e:
                    print(f"✗ Failed with providers {provider_set}: {e}")
                    if i < len(provider_options) - 1:
                        print("Trying next provider option...")
                        continue
                    else:
                        # If all provider options fail, raise an error
                        print("ONNX MODEL COMPATIBILITY ISSUE")
                        print("="*60)
                        print("Your ONNX models appear to be incompatible with the available providers.")
                        print("This often happens when models are created with specific hardware optimizations.")
                        print("\nTo fix this, you can:")
                        print("1. Download standard Whisper-Small ONNX models from Hugging Face")
                        print("2. Or convert your models to be compatible with CPU execution")
                        print("3. Install additional ONNX providers (e.g., onnxruntime-openvino for ARM)")
                        raise RuntimeError(f"Failed to load ONNX models with any available provider. Last error: {e}")
            
            print("ONNX models loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def load_audio(self, audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
        """
        Load and preprocess audio file using librosa (optimized for ONNX).
        
        Args:
            audio_path: Path to the audio file
            target_sr: Target sample rate (Whisper expects 16kHz)
            
        Returns:
            np.ndarray: Audio waveform
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio using librosa (more efficient for ONNX)
        audio, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        
        return audio
    
    def transcribe_audio(self, audio_path: Union[str, Path], 
                        language: str = "english",
                        task: str = "transcribe") -> str:
        """
        Transcribe audio file to text using ONNX models.
        
        Args:
            audio_path: Path to the audio file
            language: Language of the audio (e.g., "english", "spanish", "french")
            task: Task type ("transcribe" or "translate")
            
        Returns:
            str: Transcribed text
        """
        try:
            # Load audio
            print(f"Loading audio from: {audio_path}")
            audio = self.load_audio(audio_path)
            
            # Transcribe using ONNX models
            return self._transcribe_with_onnx(audio, language, task)
            
        except Exception as e:
            print(f"Error during transcription: {e}")
            raise
    
    def _transcribe_with_onnx(self, audio: np.ndarray, language: str, task: str) -> str:
        """Transcribe using ONNX models."""
        # Process audio using feature extractor
        print("Processing audio...")
        input_features = self.processor.feature_extractor(
            audio, 
            sampling_rate=16000, 
            return_tensors="np"
        ).input_features
        
        # Run encoder
        print("Running encoder...")
        encoder_outputs = self.encoder_session.run(
            None, 
            {"input_features": input_features}
        )
        
        # Prepare decoder inputs
        decoder_input_ids = self.processor.tokenizer(
            f"<|startoftranscript|><|{language}|><|{task}|><|notimestamps|>",
            return_tensors="np"
        ).input_ids
        
        # Run decoder (simplified greedy decoding)
        print("Running decoder...")
        generated_ids = self._decode_audio(encoder_outputs[0], decoder_input_ids)
        
        # Decode transcription
        transcription = self.processor.tokenizer.decode(
            generated_ids[0], 
            skip_special_tokens=True
        )
        
        print("Transcription completed!")
        return transcription.strip()
    
    def _decode_audio(self, encoder_outputs: np.ndarray, decoder_input_ids: np.ndarray) -> np.ndarray:
        """
        Simplified greedy decoding for ONNX model.
        """
        max_length = 448
        current_ids = decoder_input_ids.copy()
        
        for _ in range(max_length - decoder_input_ids.shape[1]):
            # Prepare inputs for decoder
            decoder_inputs = {
                "input_ids": current_ids,
                "encoder_hidden_states": encoder_outputs
            }
            
            # Run decoder
            decoder_outputs = self.decoder_session.run(None, decoder_inputs)
            logits = decoder_outputs[0]
            
            # Get next token (greedy)
            next_token = np.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            
            # Stop if EOS token
            if next_token[0, 0] == self.processor.tokenizer.eos_token_id:
                break
            
            # Append next token
            current_ids = np.concatenate([current_ids, next_token], axis=1)
        
        return current_ids


def transcribe_audio_file(audio_path: str, model_path: str, 
                         language: str = "english", 
                         task: str = "transcribe") -> str:
    """
    Convenience function to transcribe a single audio file using ONNX Whisper.
    Supports multiple audio formats: mp3, wav, m4a, flac, ogg, aac, wma, etc.
    
    Args:
        audio_path: Path to the audio file (any supported format)
        model_path: Path to the Whisper ONNX model directory
        language: Language of the audio
        task: Task type ("transcribe" or "translate")
        
    Returns:
        str: Transcribed text
    """
    transcriber = WhisperONNXTranscriber(model_path)
    return transcriber.transcribe_audio(audio_path, language, task)

# Keep the old function name for backward compatibility
def transcribe_mp3_file(mp3_path: str, model_path: str, 
                       language: str = "english", 
                       task: str = "transcribe") -> str:
    """
    Legacy function name - use transcribe_audio_file instead.
    Transcribes MP3 and other audio formats.
    """
    return transcribe_audio_file(mp3_path, model_path, language, task)


def transcribe_audio_directory(directory_path: str, model_path: str,
                             language: str = "english",
                             task: str = "transcribe") -> List[str]:
    """
    Transcribe all audio files in a directory using ONNX Whisper.
    Supports multiple audio formats: mp3, wav, m4a, flac, ogg, aac, wma, etc.
    
    Args:
        directory_path: Path to directory containing audio files
        model_path: Path to the Whisper ONNX model directory
        language: Language of the audio files
        task: Task type ("transcribe" or "translate")
        
    Returns:
        List[str]: List of transcriptions for each audio file
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Define supported audio file extensions
    audio_extensions = {
        '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', 
        '.opus', '.3gp', '.amr', '.au', '.ra', '.aiff', '.caf',
        '.mp4', '.m4b', '.m4p', '.m4r', '.mov', '.3g2', '.ac3',
        '.eac3', '.ec3', '.ac4', '.av1', '.av3'
    }
    
    # Find all audio files
    audio_files = []
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    if not audio_files:
        print(f"No audio files found in directory: {directory}")
        print(f"Supported formats: {', '.join(sorted(audio_extensions))}")
        return []
    
    # Group files by extension for display
    files_by_ext = {}
    for file_path in audio_files:
        ext = file_path.suffix.lower()
        if ext not in files_by_ext:
            files_by_ext[ext] = []
        files_by_ext[ext].append(file_path)
    
    print(f"Found {len(audio_files)} audio file(s):")
    for ext, files in sorted(files_by_ext.items()):
        print(f"  {ext.upper()}: {len(files)} file(s)")
    
    transcriber = WhisperONNXTranscriber(model_path)
    transcriptions = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing file {i}/{len(audio_files)}: {audio_file.name}")
        try:
            transcription = transcriber.transcribe_audio(audio_file, language, task)
            transcriptions.append(transcription)
        except Exception as e:
            print(f"Failed to transcribe {audio_file}: {e}")
            transcriptions.append("")  # Add empty string for failed transcription
    
    return transcriptions

# Keep the old function name for backward compatibility
def transcribe_mp3_directory(directory_path: str, model_path: str,
                           language: str = "english",
                           task: str = "transcribe") -> List[str]:
    """
    Legacy function name - use transcribe_audio_directory instead.
    Transcribes all audio formats, not just MP3.
    """
    return transcribe_audio_directory(directory_path, model_path, language, task)


# Example usage
if __name__ == "__main__":
    # Example paths - change these to match your setup
    target_directory = r"C:\Users\akarsh\Downloads"
    whisper_onnx_model_path = r"C:\Users\akarsh\Downloads\models"  # Update this path
    test_audio_path = r"C:\Users\akarsh\Downloads\Jay St.M4A"  # Update this path (supports any audio format)
    
    try:
        print("Starting document loader script...")
        print(f"Target directory: {target_directory}")
        print(f"Whisper model path: {whisper_onnx_model_path}")
        print(f"Test audio file: {test_audio_path}")
        
        # Check if paths exist
        if not Path(target_directory).exists():
            print(f"WARNING: Target directory does not exist: {target_directory}")
        else:
            print(f"✓ Target directory exists")
            
        if not Path(whisper_onnx_model_path).exists():
            print(f"WARNING: Whisper model path does not exist: {whisper_onnx_model_path}")
        else:
            print(f"✓ Whisper model path exists")
            
        if not Path(test_audio_path).exists():
            print(f"WARNING: Test audio file does not exist: {test_audio_path}")
        else:
            print(f"✓ Test audio file exists")
        
        # PDF Processing Example
        print("\n" + "="*60)
        print("PDF PROCESSING")
        print("="*60)
        
        pdf_document_lists = load_pdfs_from_directory(target_directory)
        
        if pdf_document_lists:
            print(f"Total PDFs processed: {len(pdf_document_lists)}")
            
            for i, pdf_documents in enumerate(pdf_document_lists):
                if pdf_documents:  # Check if this PDF has documents
                    first_page = pdf_documents[0]
                    pdf_name = first_page.metadata['pdf_name']
                    
                    print(f"\nPDF {i+1}: {pdf_name}")
                    print(f"  Pages: {len(pdf_documents)}")
                    print(f"  File size: {first_page.metadata['file_size']:,} bytes")
                    print(f"  Modified: {first_page.metadata['modified_date']}")
                    print(f"  Source: {first_page.metadata['source']}")
                    
                    # Show first few pages as example
                    for j, page in enumerate(pdf_documents[:3]):
                        print(f"    Page {page.metadata['page_number']}/{page.metadata['total_pages']}: {page.page_content[:100]}...")
                    
                    if len(pdf_documents) > 3:
                        print(f"    ... and {len(pdf_documents) - 3} more pages")
                else:
                    print(f"\nPDF {i+1}: Failed to extract content")
        else:
            print("No PDFs were processed.")
        
        # Audio Transcription Example
        print("\n" + "="*60)
        print("AUDIO TRANSCRIPTION")
        print("="*60)
        
        # Method 1: Transcribe a single audio file (supports multiple formats)
        print("Transcribing single audio file...")
        transcription = transcribe_audio_file(
            audio_path=test_audio_path,
            model_path=whisper_onnx_model_path,
            language="english",
            task="transcribe"
        )
        print(f"Transcription: {transcription}")
        
        # Method 2: Transcribe all audio files in a directory (multiple formats)
        print("\nTranscribing all audio files in directory...")
        transcriptions = transcribe_audio_directory(
            directory_path=target_directory,
            model_path=whisper_onnx_model_path,
            language="english",
            task="transcribe"
        )
        
        for i, transcription in enumerate(transcriptions, 1):
            print(f"\nTranscription {i}:")
            print(transcription)
        
        # Method 3: Using the class directly
        print("\nUsing WhisperONNXTranscriber class directly...")
        transcriber = WhisperONNXTranscriber(whisper_onnx_model_path)
        transcription = transcriber.transcribe_audio(
            test_audio_path,
            language="english",
            task="transcribe"
        )
        print(f"Transcription: {transcription}")
        
        # Method 4: Legacy function names (still work)
        print("\nUsing legacy function names...")
        transcription_legacy = transcribe_mp3_file(
            mp3_path=test_audio_path,
            model_path=whisper_onnx_model_path,
            language="english",
            task="transcribe"
        )
        print(f"Legacy function transcription: {transcription_legacy}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease update the paths in the example to match your setup:")
        print(f"- Target directory: {target_directory}")
        print(f"- Whisper ONNX model path: {whisper_onnx_model_path}")
        print(f"- Test audio file: {test_audio_path}")
        print("\nSupported audio formats:")
        print("MP3, WAV, M4A, FLAC, OGG, AAC, WMA, OPUS, 3GP, AMR, AU, RA, AIFF, CAF")
        print("MP4, M4B, M4P, M4R, MOV, 3G2, AC3, EAC3, EC3, AC4, AV1, AV3")
        print("\nRequired dependencies:")
        print("pip install onnxruntime transformers librosa numpy")
        print("\nFor Snapdragon X Elite optimization, also install:")
        print("pip install onnxruntime-openvino")
