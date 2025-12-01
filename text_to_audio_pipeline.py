"""
Text-to-Audio Generation Pipeline using Diffusion Models

This script implements a complete blueprint for a text-to-audio generation pipeline
using modern diffusion model concepts. It includes:
- Preprocessing: Converting audio to Mel Spectrogram
- Model Logic: Simplified U-Net architecture for noise prediction
- Post-processing: Converting Mel Spectrogram back to audio waveform
- Demonstration: Simulated generation loop with text prompt

Author: Generative AI for Music and Sound Project
License: MIT
"""

import math
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer


# =============================================================================
# CONFIGURATION
# =============================================================================

class AudioConfig:
    """Configuration for audio processing parameters."""
    SAMPLE_RATE: int = 22050  # Standard audio sample rate
    N_FFT: int = 2048  # FFT window size
    HOP_LENGTH: int = 512  # Hop length for STFT
    N_MELS: int = 128  # Number of Mel frequency bins
    DURATION: float = 5.0  # Audio duration in seconds
    
    # Diffusion parameters
    NUM_TIMESTEPS: int = 1000  # Number of diffusion timesteps
    BETA_START: float = 0.0001  # Starting beta for noise schedule
    BETA_END: float = 0.02  # Ending beta for noise schedule


# =============================================================================
# STEP A: PREPROCESSING - Audio to Mel Spectrogram
# =============================================================================

def load_audio(file_path: str, sr: int = AudioConfig.SAMPLE_RATE) -> np.ndarray:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        file_path: Path to the .wav audio file
        sr: Target sample rate
        
    Returns:
        Audio waveform as numpy array
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio


def audio_to_mel_spectrogram(
    audio: np.ndarray,
    sr: int = AudioConfig.SAMPLE_RATE,
    n_fft: int = AudioConfig.N_FFT,
    hop_length: int = AudioConfig.HOP_LENGTH,
    n_mels: int = AudioConfig.N_MELS
) -> np.ndarray:
    """
    Convert raw audio waveform to Mel Spectrogram.
    
    This is Step A of the audio generation pipeline. The Mel Spectrogram
    provides an image-like representation of sound that diffusion models
    can learn to generate.
    
    Args:
        audio: Raw audio waveform as numpy array
        sr: Sample rate of the audio
        n_fft: FFT window size
        hop_length: Number of samples between frames
        n_mels: Number of Mel frequency bins
        
    Returns:
        Mel Spectrogram as 2D numpy array (n_mels x time_frames)
    """
    # Compute Mel Spectrogram using librosa
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to log scale (dB) for better dynamic range
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range for neural network processing
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_normalized


def preprocess_audio_file(file_path: str) -> torch.Tensor:
    """
    Complete preprocessing pipeline: Load audio and convert to Mel Spectrogram tensor.
    
    Args:
        file_path: Path to the .wav audio file
        
    Returns:
        Mel Spectrogram as PyTorch tensor with shape (1, 1, n_mels, time_frames)
    """
    audio = load_audio(file_path)
    mel_spec = audio_to_mel_spectrogram(audio)
    
    # Convert to PyTorch tensor and add batch and channel dimensions
    mel_tensor = torch.from_numpy(mel_spec).float()
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, time_frames)
    
    return mel_tensor


def create_simulated_mel_spectrogram(
    duration: float = AudioConfig.DURATION,
    sr: int = AudioConfig.SAMPLE_RATE,
    n_mels: int = AudioConfig.N_MELS,
    hop_length: int = AudioConfig.HOP_LENGTH
) -> torch.Tensor:
    """
    Create a simulated Mel Spectrogram for demonstration purposes.
    
    Args:
        duration: Duration in seconds
        sr: Sample rate
        n_mels: Number of Mel bins
        hop_length: Hop length
        
    Returns:
        Simulated Mel Spectrogram tensor
    """
    # Calculate number of time frames
    n_samples = int(duration * sr)
    n_frames = n_samples // hop_length + 1
    
    # Create a simulated spectrogram with some structure
    t = np.linspace(0, duration, n_frames)
    f = np.linspace(0, 1, n_mels)
    
    # Create harmonic patterns to simulate music
    mel_spec = np.zeros((n_mels, n_frames))
    for i, freq in enumerate([0.1, 0.2, 0.3, 0.5]):
        harmonic = np.sin(2 * np.pi * freq * t)
        mel_spec += np.outer(np.exp(-(f - freq)**2 / 0.01), harmonic)
    
    # Normalize
    mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
    
    mel_tensor = torch.from_numpy(mel_spec).float()
    return mel_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_mels, n_frames)


# =============================================================================
# STEP B: MODEL LOGIC - Simplified U-Net for Noise Prediction
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for diffusion timestep encoding.
    
    This embedding helps the model understand at which step of the
    diffusion process it currently is.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal embeddings for timesteps.
        
        Args:
            time: Tensor of timesteps with shape (batch_size,)
            
        Returns:
            Embeddings with shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """
    Convolutional block with GroupNorm and activation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block for U-Net encoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with skip connection output.
        
        Args:
            x: Input tensor
            time_emb: Time embedding tensor
            
        Returns:
            Tuple of (downsampled output, skip connection)
        """
        x = self.conv1(x)
        x = x + self.time_mlp(time_emb)[:, :, None, None]
        x = self.conv2(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block for U-Net decoder.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBlock(in_channels + out_channels, out_channels)  # Concatenate skip connection
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection input.
        
        Args:
            x: Input tensor from previous layer
            skip: Skip connection from encoder
            time_emb: Time embedding tensor
            
        Returns:
            Upsampled output tensor
        """
        x = self.upsample(x)
        
        # Handle size mismatch between upsampled tensor and skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = x + self.time_mlp(time_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class MiddleBlock(nn.Module):
    """
    Middle block of U-Net with attention mechanism.
    """
    
    def __init__(self, channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        
        # Simplified attention
        self.attention = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.norm = nn.GroupNorm(min(8, channels), channels)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x + self.time_mlp(time_emb)[:, :, None, None]
        
        # Apply attention
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1)  # (b, hw, c)
        x_attn, _ = self.attention(x_flat, x_flat, x_flat)
        x_attn = x_attn.permute(0, 2, 1).view(b, c, h, w)
        x = self.norm(x + x_attn)
        
        x = self.conv2(x)
        return x


class TextConditioningModule(nn.Module):
    """
    Module to condition the U-Net on text embeddings.
    
    Uses cross-attention to inject text information into the generation process.
    """
    
    def __init__(self, text_embed_dim: int, model_dim: int):
        super().__init__()
        self.projection = nn.Linear(text_embed_dim, model_dim)
        self.norm = nn.LayerNorm(model_dim)
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to model dimension.
        
        Args:
            text_embeddings: Text embeddings from CLIP
            
        Returns:
            Projected embeddings
        """
        x = self.projection(text_embeddings)
        x = self.norm(x)
        return x


class SimplifiedAudioUNet(nn.Module):
    """
    Simplified U-Net architecture for audio diffusion.
    
    This is the core noise prediction model used in diffusion-based
    audio generation. It predicts the noise added to a Mel Spectrogram
    at each diffusion timestep.
    
    Architecture:
    - Encoder: Series of DownBlocks that compress the input
    - Middle: Processing block with attention
    - Decoder: Series of UpBlocks that expand back to original size
    - Skip connections between encoder and decoder
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 256,
        text_embed_dim: int = 512
    ):
        """
        Initialize the U-Net model.
        
        Args:
            in_channels: Number of input channels (1 for Mel Spectrogram)
            out_channels: Number of output channels
            base_channels: Base number of channels
            channel_multipliers: Multipliers for each level
            time_emb_dim: Dimension of time embedding
            text_embed_dim: Dimension of text conditioning embedding
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        
        # Text conditioning
        self.text_conditioning = TextConditioningModule(text_embed_dim, time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder (Downsampling path)
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for mult in channel_multipliers:
            out_ch = base_channels * mult
            self.down_blocks.append(DownBlock(current_channels, out_ch, time_emb_dim))
            channels.append(out_ch)
            current_channels = out_ch
        
        # Middle block
        self.middle_block = MiddleBlock(current_channels, time_emb_dim)
        
        # Decoder (Upsampling path)
        self.up_blocks = nn.ModuleList()
        
        for mult in reversed(channel_multipliers):
            out_ch = base_channels * mult
            self.up_blocks.append(UpBlock(current_channels, out_ch, time_emb_dim))
            current_channels = out_ch
        
        # Final convolution
        self.final_conv = nn.Sequential(
            ConvBlock(current_channels, current_channels),
            nn.Conv2d(current_channels, out_channels, kernel_size=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the U-Net.
        
        Args:
            x: Noisy Mel Spectrogram tensor (batch, channels, height, width)
            timestep: Current diffusion timestep (batch,)
            text_embedding: Optional text conditioning embedding
            
        Returns:
            Predicted noise tensor with same shape as input
        """
        # Compute time embedding
        time_emb = self.time_embedding(timestep)
        
        # Add text conditioning if provided
        if text_embedding is not None:
            # Average pool the sequence dimension
            if text_embedding.dim() == 3:
                text_emb = text_embedding.mean(dim=1)
            else:
                text_emb = text_embedding
            text_cond = self.text_conditioning(text_emb)
            time_emb = time_emb + text_cond
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, time_emb)
            skips.append(skip)
        
        # Middle block
        x = self.middle_block(x, time_emb)
        
        # Decoder with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip, time_emb)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x


# =============================================================================
# DIFFUSION PROCESS
# =============================================================================

class GaussianDiffusion:
    """
    Gaussian Diffusion process for audio generation.
    
    Implements the forward (adding noise) and reverse (removing noise)
    diffusion processes.
    """
    
    def __init__(
        self,
        num_timesteps: int = AudioConfig.NUM_TIMESTEPS,
        beta_start: float = AudioConfig.BETA_START,
        beta_end: float = AudioConfig.BETA_END
    ):
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion process: Add noise to the input.
        
        Args:
            x_start: Original clean input
            noise: Random Gaussian noise
            timesteps: Timesteps for each sample in batch
            
        Returns:
            Noisy input at timestep t
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha[:, None, None, None]
        sqrt_one_minus_alpha = sqrt_one_minus_alpha[:, None, None, None]
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def remove_noise(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """
        Reverse diffusion process: Remove noise from the input.
        
        Uses the predicted noise to denoise the input at timestep t.
        
        Args:
            x_t: Noisy input at timestep t
            predicted_noise: Noise predicted by the model
            timestep: Current timestep
            
        Returns:
            Denoised input (x_{t-1})
        """
        alpha = self.alphas[timestep]
        alpha_cumprod = self.alphas_cumprod[timestep]
        beta = self.betas[timestep]
        
        # Predict x_0
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timestep]
        sqrt_recip_alpha = 1 / torch.sqrt(alpha)
        
        # Mean of the posterior distribution
        model_mean = sqrt_recip_alpha * (
            x_t - beta * predicted_noise / sqrt_one_minus_alpha_cumprod
        )
        
        # Add noise for all timesteps except t=0
        if timestep > 0:
            noise = torch.randn_like(x_t)
            posterior_variance = self.posterior_variance[timestep]
            model_mean = model_mean + torch.sqrt(posterior_variance) * noise
        
        return model_mean


# =============================================================================
# STEP C: POST-PROCESSING - Mel Spectrogram to Audio
# =============================================================================

def mel_spectrogram_to_audio(
    mel_spec: np.ndarray,
    sr: int = AudioConfig.SAMPLE_RATE,
    n_fft: int = AudioConfig.N_FFT,
    hop_length: int = AudioConfig.HOP_LENGTH,
    n_mels: int = AudioConfig.N_MELS,
    n_iter: int = 32
) -> np.ndarray:
    """
    Convert Mel Spectrogram back to audio waveform using Griffin-Lim algorithm.
    
    This is Step C of the audio generation pipeline. It reconstructs
    the audio waveform from the generated Mel Spectrogram.
    
    Args:
        mel_spec: Mel Spectrogram as 2D numpy array (n_mels x time_frames)
        sr: Target sample rate
        n_fft: FFT window size (must match preprocessing)
        hop_length: Hop length (must match preprocessing)
        n_mels: Number of Mel bins (must match preprocessing)
        n_iter: Number of Griffin-Lim iterations
        
    Returns:
        Audio waveform as numpy array
    """
    # Denormalize from [0, 1] to dB scale (approximate range)
    mel_spec_db = mel_spec * 80.0 - 80.0  # Approximate dB range
    
    # Convert from dB to power
    mel_spec_power = librosa.db_to_power(mel_spec_db)
    
    # Create Mel filterbank
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    
    # Invert Mel filterbank (pseudo-inverse)
    mel_basis_inv = np.linalg.pinv(mel_basis)
    
    # Convert Mel spectrogram to linear spectrogram
    linear_spec = np.maximum(1e-10, np.dot(mel_basis_inv, mel_spec_power))
    
    # Use Griffin-Lim algorithm to reconstruct audio
    audio = librosa.griffinlim(
        linear_spec,
        n_iter=n_iter,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    return audio


def postprocess_mel_tensor(mel_tensor: torch.Tensor) -> np.ndarray:
    """
    Complete post-processing pipeline: Convert Mel Spectrogram tensor to audio.
    
    Args:
        mel_tensor: Mel Spectrogram tensor with shape (batch, channels, n_mels, time)
        
    Returns:
        Audio waveform as numpy array
    """
    # Remove batch and channel dimensions
    mel_spec = mel_tensor.squeeze().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    mel_spec = np.clip(mel_spec, 0, 1)
    
    # Convert to audio
    audio = mel_spectrogram_to_audio(mel_spec)
    
    return audio


def save_audio(audio: np.ndarray, file_path: str, sr: int = AudioConfig.SAMPLE_RATE):
    """
    Save audio waveform to a .wav file.
    
    Args:
        audio: Audio waveform as numpy array
        file_path: Output file path
        sr: Sample rate
    """
    # Normalize audio to prevent clipping
    audio = audio / np.max(np.abs(audio) + 1e-8)
    sf.write(file_path, audio, sr)


# =============================================================================
# TEXT ENCODING
# =============================================================================

class TextEncoder:
    """
    Text encoder using CLIP for conditioning the diffusion model.
    
    This class handles the encoding of text prompts into embeddings
    that can be used to condition the audio generation process.
    
    Note: When CLIP model is not available (e.g., in offline environments),
    it falls back to simulated embeddings for demonstration purposes.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", use_real_clip: bool = False):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Name of the CLIP model to use
            use_real_clip: Whether to attempt loading the real CLIP model.
                          Set to False for offline/demo mode (default).
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._initialized = False
        self._use_real_clip = use_real_clip
    
    def _lazy_init(self):
        """Lazily initialize the model to avoid loading unless needed."""
        if not self._initialized:
            if self._use_real_clip:
                try:
                    # Only attempt to load CLIP if explicitly requested
                    import os
                    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
                    os.environ['TRANSFORMERS_OFFLINE'] = '0'
                    
                    self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
                    self.model = CLIPTextModel.from_pretrained(self.model_name)
                    self.model.eval()
                    self._initialized = True
                    print("   - Loaded real CLIP model for text encoding")
                except Exception as e:
                    print(f"   - Note: CLIP model unavailable, using simulated embeddings")
                    self._initialized = False
            else:
                # Default to simulated embeddings (faster, works offline)
                print("   - Using simulated text embeddings (demo mode)")
                self._initialized = False
    
    def encode(self, text: str, max_length: int = 77) -> torch.Tensor:
        """
        Encode text prompt into embeddings.
        
        Args:
            text: Text prompt to encode
            max_length: Maximum token length
            
        Returns:
            Text embeddings tensor
        """
        self._lazy_init()
        
        if self.tokenizer is not None and self.model is not None:
            # Real CLIP encoding
            tokens = self.tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = self.model(**tokens)
                embeddings = outputs.last_hidden_state
            
            return embeddings
        else:
            # Simulated embeddings for demonstration
            return self._simulate_embeddings(text)
    
    def _simulate_embeddings(self, text: str) -> torch.Tensor:
        """
        Create simulated text embeddings for demonstration.
        
        The embeddings are deterministic based on the text content,
        ensuring reproducible results for the same prompt.
        
        Args:
            text: Text prompt
            
        Returns:
            Simulated embeddings tensor with shape (1, 77, 512)
        """
        # Use text hash to create deterministic but varied embeddings
        text_hash = hash(text)
        np.random.seed(abs(text_hash) % (2**31))
        
        # Create embeddings with shape (1, sequence_length, embed_dim)
        embeddings = np.random.randn(1, 77, 512).astype(np.float32)
        
        return torch.from_numpy(embeddings)


# =============================================================================
# DEMONSTRATION: GENERATION LOOP
# =============================================================================

def simulated_generation_demo(
    prompt: str = "A calm ambient electronic music with soft pads",
    num_inference_steps: int = 50,
    use_real_clip: bool = False
):
    """
    Demonstrate the complete text-to-audio generation pipeline.
    
    This function shows the entire process:
    1. Encode the text prompt
    2. Start with pure noise
    3. Iteratively denoise using the U-Net
    4. Convert the final Mel Spectrogram to audio
    
    Args:
        prompt: Text description of the desired audio
        num_inference_steps: Number of denoising steps
        use_real_clip: Whether to attempt loading real CLIP model
    """
    print("=" * 70)
    print("TEXT-TO-AUDIO GENERATION PIPELINE DEMONSTRATION")
    print("=" * 70)
    print(f"\nPrompt: '{prompt}'")
    print(f"Inference steps: {num_inference_steps}")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize components
    print("\n[1/6] Initializing components...")
    
    # Text encoder
    text_encoder = TextEncoder()
    
    # U-Net model
    model = SimplifiedAudioUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,  # Reduced for demonstration
        channel_multipliers=(1, 2, 4),
        time_emb_dim=128,
        text_embed_dim=512
    ).to(device)
    
    # Diffusion process
    diffusion = GaussianDiffusion(num_timesteps=1000)
    
    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Encode text prompt
    print("\n[2/6] Encoding text prompt...")
    text_embedding = text_encoder.encode(prompt)
    text_embedding = text_embedding.to(device)
    print(f"   - Text embedding shape: {text_embedding.shape}")
    
    # Create initial noise (representing the target Mel Spectrogram shape)
    print("\n[3/6] Creating initial noise...")
    n_mels = AudioConfig.N_MELS
    duration = AudioConfig.DURATION
    sr = AudioConfig.SAMPLE_RATE
    hop_length = AudioConfig.HOP_LENGTH
    n_frames = int(duration * sr / hop_length) + 1
    
    # Start with pure Gaussian noise
    x_t = torch.randn(1, 1, n_mels, n_frames).to(device)
    print(f"   - Initial noise shape: {x_t.shape}")
    print(f"   - Initial noise mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
    
    # Denoising loop
    print("\n[4/6] Running denoising loop...")
    
    # Use a subset of timesteps for faster inference
    timesteps = torch.linspace(999, 0, num_inference_steps, dtype=torch.long)
    
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = t.unsqueeze(0).to(device)
            
            # Predict noise
            predicted_noise = model(x_t, t_tensor.float(), text_embedding)
            
            # Remove noise (reverse diffusion step)
            t_int = int(t.item())
            x_t = diffusion.remove_noise(x_t, predicted_noise, t_int)
            
            # Progress update
            if (i + 1) % 10 == 0 or i == 0:
                print(f"   - Step {i + 1}/{num_inference_steps}, "
                      f"t={t_int}, "
                      f"mel mean: {x_t.mean().item():.4f}, "
                      f"std: {x_t.std().item():.4f}")
    
    # Post-process: normalize to [0, 1]
    print("\n[5/6] Post-processing generated spectrogram...")
    generated_mel = x_t.squeeze().cpu()
    
    # Normalize to [0, 1] range
    generated_mel = (generated_mel - generated_mel.min()) / (generated_mel.max() - generated_mel.min() + 1e-8)
    print(f"   - Generated Mel shape: {generated_mel.shape}")
    print(f"   - Value range: [{generated_mel.min().item():.4f}, {generated_mel.max().item():.4f}]")
    
    # Convert to audio
    print("\n[6/6] Converting Mel Spectrogram to audio...")
    try:
        audio = mel_spectrogram_to_audio(generated_mel.numpy())
        print(f"   - Generated audio length: {len(audio) / sr:.2f} seconds")
        print(f"   - Audio samples: {len(audio)}")
        
        # Note: In a real scenario, you would save the audio
        # save_audio(audio, "generated_audio.wav")
        print("   - Audio ready for playback (not saved in demo)")
    except Exception as e:
        print(f"   - Audio conversion note: {e}")
        print("   - (This is expected with random model weights)")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nThis demo showed the complete pipeline:")
    print("  A. Preprocessing: Audio → Mel Spectrogram (simulated)")
    print("  B. Model: U-Net noise prediction with text conditioning")
    print("  C. Post-processing: Mel Spectrogram → Audio waveform")
    print("\nNote: The model is not trained, so output is random.")
    print("With training on audio data, it would generate coherent sounds.")
    print()
    
    return generated_mel


def demonstrate_preprocessing():
    """
    Demonstrate the preprocessing step (Step A).
    
    Shows how to convert audio to Mel Spectrogram.
    """
    print("\n" + "=" * 70)
    print("STEP A: PREPROCESSING DEMONSTRATION")
    print("=" * 70)
    
    # Create a simulated audio signal (sine wave)
    print("\nCreating simulated audio signal...")
    sr = AudioConfig.SAMPLE_RATE
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Multi-frequency test signal
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 220 * t)    # A3 note
    )
    
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Audio shape: {audio.shape}")
    
    # Convert to Mel Spectrogram
    print("\nConverting to Mel Spectrogram...")
    mel_spec = audio_to_mel_spectrogram(audio, sr=sr)
    
    print(f"   - Mel Spectrogram shape: {mel_spec.shape}")
    print(f"   - Value range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"   - Mel bins: {mel_spec.shape[0]}")
    print(f"   - Time frames: {mel_spec.shape[1]}")
    
    # Convert to tensor
    mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0)
    print(f"\n   - Tensor shape for model: {mel_tensor.shape}")
    
    return mel_spec


def demonstrate_postprocessing(mel_spec: Optional[np.ndarray] = None):
    """
    Demonstrate the post-processing step (Step C).
    
    Shows how to convert Mel Spectrogram back to audio.
    """
    print("\n" + "=" * 70)
    print("STEP C: POST-PROCESSING DEMONSTRATION")
    print("=" * 70)
    
    if mel_spec is None:
        # Create a simulated Mel Spectrogram
        print("\nCreating simulated Mel Spectrogram...")
        mel_tensor = create_simulated_mel_spectrogram(duration=2.0)
        mel_spec = mel_tensor.squeeze().numpy()
    
    print(f"   - Input Mel Spectrogram shape: {mel_spec.shape}")
    
    # Convert to audio
    print("\nConverting Mel Spectrogram to audio waveform...")
    audio = mel_spectrogram_to_audio(mel_spec)
    
    sr = AudioConfig.SAMPLE_RATE
    print(f"   - Output audio shape: {audio.shape}")
    print(f"   - Duration: {len(audio) / sr:.2f} seconds")
    print(f"   - Sample rate: {sr} Hz")
    
    return audio


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the text-to-audio generation pipeline.
    
    Demonstrates all three steps of the pipeline:
    A. Preprocessing: Audio to Mel Spectrogram
    B. Model: U-Net architecture for noise prediction
    C. Post-processing: Mel Spectrogram to Audio
    """
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  TEXT-TO-AUDIO GENERATION PIPELINE".center(68) + "*")
    print("*" + "  Using Diffusion Models".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    # Demonstrate preprocessing (Step A)
    mel_spec = demonstrate_preprocessing()
    
    # Demonstrate the model architecture (Step B) and generation
    print("\n" + "=" * 70)
    print("STEP B: U-NET MODEL ARCHITECTURE")
    print("=" * 70)
    
    model = SimplifiedAudioUNet(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        time_emb_dim=128,
        text_embed_dim=512
    )
    
    print("\nModel Summary:")
    print(f"   - Input channels: 1")
    print(f"   - Output channels: 1")
    print(f"   - Base channels: 32")
    print(f"   - Channel multipliers: (1, 2, 4)")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nModel Architecture:")
    print("   - Encoder: 3 DownBlocks with skip connections")
    print("   - Middle: Attention-based processing block")
    print("   - Decoder: 3 UpBlocks with skip connections")
    print("   - Time embedding: Sinusoidal position encoding")
    print("   - Text conditioning: CLIP-based cross-attention")
    
    # Demonstrate post-processing (Step C)
    demonstrate_postprocessing(mel_spec)
    
    # Run the full generation demo
    simulated_generation_demo(
        prompt="Peaceful ambient music with soft synthesizer pads",
        num_inference_steps=20  # Reduced for faster demo
    )
    
    print("\n" + "*" * 70)
    print("Pipeline demonstration complete!")
    print("*" * 70 + "\n")


if __name__ == "__main__":
    main()
