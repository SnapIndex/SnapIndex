# ---------------------------------------------------------------------
# Standalone Whisper implementation without AI Hub dependencies
# Based on original Qualcomm AI Hub models but simplified for PyInstaller
# ---------------------------------------------------------------------
import numpy as np
import os
import torch
import whisper
import librosa
from scipy import special as scipy_special


# Whisper constants
SAMPLE_RATE = 16000
CHUNK_LENGTH = 30
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000
AUDIO_EMB_LEN = int(N_SAMPLES / N_MELS / 4)  # 1500
MELS_AUDIO_LEN = AUDIO_EMB_LEN * 2  # 3000
MEAN_DECODE_LEN = 224

# Token constants
TOKEN_SOT = 50257  # Start of transcript
TOKEN_EOT = 50256  # end of transcript
TOKEN_BLANK = 220  # " "
TOKEN_NO_TIMESTAMP = 50362
TOKEN_TIMESTAMP_BEGIN = 50363
TOKEN_NO_SPEECH = 50361
NO_SPEECH_THR = 0.6

# Non-speech tokens to suppress
NON_SPEECH_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
]


class TorchNumpyAdapter:
    """Simple adapter to convert between torch tensors and numpy arrays"""
    def __init__(self, model):
        self.model = model
    
    def __call__(self, *args, **kwargs):
        # Convert numpy inputs to torch tensors
        torch_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                torch_args.append(torch.from_numpy(arg))
            else:
                torch_args.append(arg)
        
        torch_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                torch_kwargs[key] = torch.from_numpy(value)
            else:
                torch_kwargs[key] = value
        
        # Run the model
        with torch.no_grad():
            result = self.model(*torch_args, **torch_kwargs)
        
        # Convert torch outputs back to numpy
        if isinstance(result, torch.Tensor):
            return result.detach().numpy()
        elif isinstance(result, (list, tuple)):
            return [r.detach().numpy() if isinstance(r, torch.Tensor) else r for r in result]
        else:
            return result


class StandaloneWhisperApp:
    """
    Standalone WhisperApp that works without AI Hub dependencies.
    Compatible with PyInstaller executables.
    """

    def __init__(
        self,
        encoder,
        decoder, 
        num_decoder_blocks: int,
        num_decoder_heads: int,
        attention_dim: int,
        mel_filter: np.ndarray | None = None,
        sample_rate: int = SAMPLE_RATE,
        max_audio_seconds: int = CHUNK_LENGTH,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.num_decoder_blocks = num_decoder_blocks
        self.num_decoder_heads = num_decoder_heads
        self.attention_dim = attention_dim
        self.mean_decode_len = MEAN_DECODE_LEN

        # Set audio parameters first (needed for mel filter creation)
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.n_fft = n_fft
        self.max_audio_samples = self.max_audio_seconds * self.sample_rate

        # Create default mel filter if not provided (requires sample_rate to be set)
        self.mel_filter = mel_filter
        self.mel_filter_loaded = False  # Track if we've shown the loading message
        if self.mel_filter is None:
            self.mel_filter = self._load_or_create_mel_filter()

    def _load_or_create_mel_filter(self):
        """Load proper mel filters from mel_filters.npz if available, otherwise create simplified version"""
        
        # Try to load the proper Whisper mel filters first
        try:
            mel_filter_path = "mel_filters.npz"
            if os.path.exists(mel_filter_path):
                # if not self.mel_filter_loaded:
                    # print("✅ Loading optimized Whisper mel filters from mel_filters.npz")
                loaded = np.load(mel_filter_path)
                if 'mel_filters' in loaded:
                    mel_filter_matrix = loaded['mel_filters']
                    loaded.close()
                    if not self.mel_filter_loaded:
                        # print(f"   Loaded mel filter shape: {mel_filter_matrix.shape}")
                        # print(f"   Using high-quality mel filters (same as AI Hub version)")
                        self.mel_filter_loaded = True
                    return mel_filter_matrix.astype(np.float32)
                else:
                    loaded.close()
                    if not self.mel_filter_loaded:
                        print("⚠ mel_filters.npz found but doesn't contain 'mel_filters' key")
            else:
                if not self.mel_filter_loaded:
                    print("⚠ mel_filters.npz not found - using simplified mel filters")
        except Exception as e:
            if not self.mel_filter_loaded:
                print(f"⚠ Error loading mel_filters.npz: {e}")
                print("⚠ Falling back to simplified mel filters")
        
        # Fall back to simplified mel filter creation
        self.mel_filter_loaded = True
        return self._create_simplified_mel_filter()

    def _create_simplified_mel_filter(self):
        """Create a basic mel filter bank"""
        # This is a simplified mel filter - for production use, you'd want to load the proper one
        # But this works for basic functionality without external dependencies
        
        # Simple mel filter bank creation
        n_mels = N_MELS
        fmax = self.sample_rate // 2
        
        # Create mel scale
        mel_f = 2595 * np.log10(1 + np.linspace(0, fmax, n_mels + 2) / 700)
        hz_points = 700 * (10**(mel_f / 2595) - 1)
        
        # Convert to fft bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate)
        
        # Create filter bank
        fbank = np.zeros([n_mels, self.n_fft // 2 + 1])
        for j in range(0, n_mels):
            for i in range(int(bin_points[j]), int(bin_points[j + 1])):
                fbank[j, i] = (i - bin_points[j]) / (bin_points[j + 1] - bin_points[j])
            for i in range(int(bin_points[j + 1]), int(bin_points[j + 2])):
                fbank[j, i] = (bin_points[j + 2] - i) / (bin_points[j + 2] - bin_points[j + 1])
        
        return fbank.astype(np.float32)

    def transcribe(
        self, audio: np.ndarray | str, audio_sample_rate: int | None = None
    ) -> str:
        """
        Transcribe the provided audio to text.
        """
        if isinstance(audio, str):
            try:
                import audio2numpy as a2n
                audio, audio_sample_rate = a2n.audio_from_file(audio)
            except ImportError:
                raise ImportError("audio2numpy required for file input. Install with: pip install audio2numpy")

        assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)

        return " ".join(
            self._transcribe_single_chunk(x)
            for x in self._chunk_and_resample_audio(audio, audio_sample_rate)
        )

    def _transcribe_single_chunk(self, audio: np.ndarray) -> str:
        """Transcribe an audio chunk to text."""
        mel_input = self._log_mel_spectrogram(audio)
        k_cache_cross, v_cache_cross = self.encoder(mel_input)
        
        # Start decoding
        x = np.array([[TOKEN_SOT]])
        decoded_tokens = [TOKEN_SOT]
        sample_len = self.mean_decode_len
        
        k_cache_self = np.zeros((
            self.num_decoder_blocks,
            self.num_decoder_heads,
            self.attention_dim // self.num_decoder_heads,
            sample_len,
        )).astype(np.float32)
        
        v_cache_self = np.zeros((
            self.num_decoder_blocks,
            self.num_decoder_heads,
            sample_len,
            self.attention_dim // self.num_decoder_heads,
        )).astype(np.float32)

        for i in range(sample_len):
            index = np.zeros([1, 1], dtype=np.int32)
            index[0, 0] = i
            
            decoder_out = self.decoder(
                x, index, k_cache_cross, v_cache_cross, k_cache_self, v_cache_self
            )
            
            logits = decoder_out[0]
            k_cache_self = decoder_out[1]
            v_cache_self = decoder_out[2]

            # Debug: Print logits info
            if i < 3:  # Only print first few iterations to avoid spam
                print(f"🔍 Debug: Iteration {i}, logits shape: {logits.shape}")
                print(f"🔍 Debug: Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
                print(f"🔍 Debug: Top 5 logits: {np.argsort(logits[0, -1])[-5:]}")

            logits = logits[0, -1]  # consider only the last token

            # Apply filters
            if i == 0:
                logits[[TOKEN_EOT, TOKEN_BLANK]] = -np.inf
            logits[NON_SPEECH_TOKENS] = -np.inf

            logits, logprobs = self._apply_timestamp_rules(logits, decoded_tokens)

            if i == 0:
                # detect no_speech
                no_speech_prob = np.exp(logprobs[TOKEN_NO_SPEECH])
                if no_speech_prob > NO_SPEECH_THR:
                    break

            next_token = np.argmax(logits)
            if next_token == TOKEN_EOT:
                break

            x = np.array([[next_token]])
            decoded_tokens.append(int(next_token))

        # Debug: Print decoded tokens
        print(f"🔍 Debug: Decoded tokens: {decoded_tokens}")
        print(f"🔍 Debug: Number of tokens: {len(decoded_tokens)}")
        
        # Decode tokens to text
        try:
            tokenizer = whisper.decoding.get_tokenizer(
                multilingual=False, language="en", task="transcribe"
            )
            text = tokenizer.decode(decoded_tokens[1:])  # remove TOKEN_SOT
            print(f"🔍 Debug: Whisper tokenizer result: '{text}'")
            return text.strip()
        except Exception as e:
            print(f"⚠ Warning: Could not decode tokens properly: {e}")
            # Fallback: try to decode with basic method
            try:
                import tiktoken
                encoding = tiktoken.get_encoding("gpt2")
                text = encoding.decode(decoded_tokens[1:])  # remove TOKEN_SOT
                print(f"🔍 Debug: Tiktoken result: '{text}'")
                return text.strip()
            except Exception as e2:
                print(f"⚠ Warning: Fallback tokenizer also failed: {e2}")
                # Last resort: return token count info
                return f"[Decoded {len(decoded_tokens)-1} tokens but cannot convert to text]"

    def _log_mel_spectrogram(self, audio_np: np.ndarray) -> np.ndarray:
        """Compute the log-Mel spectrogram using numpy/scipy instead of PyTorch"""
        import scipy.signal as signal
        
        audio = audio_np.copy()
        
        # Pad audio if needed
        padding = self.max_audio_samples - len(audio)
        if padding > 0:
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Create Hann window
        window = signal.windows.hann(self.n_fft)
        
        # Compute STFT using scipy
        f, t, stft = signal.stft(audio, fs=self.sample_rate, window=window, 
                                nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        
        # Get magnitudes - ensure we have the right number of frequency bins
        magnitudes = np.abs(stft) ** 2
        
        # Debug: Print dimensions
        print(f"🔍 Debug: STFT shape: {stft.shape}")
        print(f"🔍 Debug: Magnitudes shape: {magnitudes.shape}")
        print(f"🔍 Debug: Mel filter shape: {self.mel_filter.shape}")
        
        # Ensure mel filter and magnitudes have compatible dimensions
        if self.mel_filter.shape[1] != magnitudes.shape[0]:
            print(f"⚠️  Dimension mismatch: mel_filter {self.mel_filter.shape[1]} vs magnitudes {magnitudes.shape[0]}")
            # Adjust mel filter to match STFT output
            if self.mel_filter.shape[1] > magnitudes.shape[0]:
                # Truncate mel filter
                self.mel_filter = self.mel_filter[:, :magnitudes.shape[0]]
                print(f"🔧 Truncated mel filter to shape: {self.mel_filter.shape}")
            else:
                # Pad magnitudes
                pad_width = self.mel_filter.shape[1] - magnitudes.shape[0]
                magnitudes = np.pad(magnitudes, ((0, pad_width), (0, 0)), mode='constant')
                print(f"🔧 Padded magnitudes to shape: {magnitudes.shape}")
        
        # Apply mel filter
        mel_spec = self.mel_filter @ magnitudes
        
        # Convert to log scale
        log_spec = np.clip(mel_spec, a_min=1e-10, a_max=None)
        log_spec = np.log10(log_spec)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        
        # Add batch dimension
        return log_spec[np.newaxis, :, :].astype(np.float32)

    def _chunk_and_resample_audio(
        self, 
        audio: np.ndarray,
        audio_sample_rate: int,
    ) -> list[np.ndarray]:
        """Chunk and resample audio"""
        if audio_sample_rate != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=audio_sample_rate, target_sr=self.sample_rate)

        number_of_full_length_audio_chunks = (
            audio.shape[0] // self.sample_rate // self.max_audio_seconds
        )
        last_sample_in_full_length_audio_chunks = (
            self.sample_rate * number_of_full_length_audio_chunks * self.max_audio_seconds
        )

        if number_of_full_length_audio_chunks == 0:
            return [audio]

        return [
            *np.array_split(
                audio[:last_sample_in_full_length_audio_chunks],
                number_of_full_length_audio_chunks,
            ),
            audio[last_sample_in_full_length_audio_chunks:],
        ]

    def _apply_timestamp_rules(
        self, logits: np.ndarray, tokens: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply timestamp rules during decoding"""
        # Simplified timestamp rules - full implementation would be more complex
        logits[TOKEN_NO_TIMESTAMP] = -np.inf
        
        # For simplicity, suppress timestamps at the beginning
        if len(tokens) == 1:  # Only SOT token
            logits[TOKEN_TIMESTAMP_BEGIN:] = -np.inf
        
        logprobs = scipy_special.log_softmax(logits)
        return logits, logprobs
