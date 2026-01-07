"""
Tiny GRU for Adaptive Alpha Control

This module trains a small GRU network to predict optimal alpha values
for the Decision-Directed SNR estimation in Log-MMSE enhancement.

The GRU learns to:
- Use low alpha (0.9) during speech onsets for fast response
- Use high alpha (0.98) during steady speech for smoothing
- Use very high alpha (0.995) during noise-only for stability

Architecture:
    Input: 5 spectral features per frame
    GRU: 32 hidden units
    Output: 1 (alpha value, sigmoid * 0.1 + 0.9 = [0.9, 1.0])

Training: CPU-only, ~10-30 minutes on NOIZEUS dataset
"""

import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# Standalone STFT (no external dependencies)
def simple_stft(audio, frame_size=256, hop_size=64):
    """Simple STFT using numpy."""
    window = np.hanning(frame_size)
    n_frames = (len(audio) - frame_size) // hop_size + 1
    n_bins = frame_size // 2 + 1
    
    spectrogram = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size] * window
        spectrogram[i] = np.fft.rfft(frame, frame_size)
    
    return spectrogram


# ============================================
# Feature Extraction
# ============================================

def extract_features(audio: np.ndarray, sample_rate: int = 8000, hop_size: Optional[int] = None) -> np.ndarray:
    """
    Extract spectral features for GRU input.
    
    Args:
        audio: Input audio array
        sample_rate: Sample rate (default 8000)
        hop_size: Hop size in samples. If None, defaults to 8ms (64 samples @ 8k).
                  Use 128 for 16ms hop (fast mode).
    
    Features per frame (5 total):
    1. Log power (normalized)
    2. Spectral flatness (speech vs noise indicator)
    3. Spectral centroid (normalized)
    4. Delta power (temporal change)
    5. Zero-crossing rate
    
    Returns:
        features: (n_frames, 5)
    """
    # Use simple STFT
    frame_size = int(0.032 * sample_rate)  # 32ms
    if hop_size is None:
        hop_size = int(0.008 * sample_rate)    # 8ms
    
    spec = simple_stft(audio, frame_size, hop_size)
    power = np.abs(spec) ** 2
    n_frames = power.shape[0]
    
    features = np.zeros((n_frames, 5))
    
    # 1. Log power (mean across frequency, normalized)
    log_power = np.log10(np.mean(power, axis=1) + 1e-10)
    log_power = (log_power - log_power.min()) / (log_power.max() - log_power.min() + 1e-10)
    features[:, 0] = log_power
    
    # 2. Spectral flatness (geometric/arithmetic mean ratio)
    geometric_mean = np.exp(np.mean(np.log(power + 1e-10), axis=1))
    arithmetic_mean = np.mean(power, axis=1)
    flatness = geometric_mean / (arithmetic_mean + 1e-10)
    features[:, 1] = np.clip(flatness, 0, 1)
    
    # 3. Spectral centroid (normalized)
    freq_bins = np.arange(power.shape[1])
    centroid = np.sum(power * freq_bins, axis=1) / (np.sum(power, axis=1) + 1e-10)
    centroid = centroid / power.shape[1]  # Normalize to [0, 1]
    features[:, 2] = centroid
    
    # 4. Delta power (temporal derivative)
    delta = np.diff(log_power, prepend=log_power[0])
    features[:, 3] = np.clip(delta, -1, 1)
    
    # 5. Simple voice activity (power above threshold)
    threshold = np.percentile(log_power, 30)
    vad = (log_power > threshold).astype(float)
    features[:, 4] = vad
    
    return features


def compute_oracle_alpha(clean: np.ndarray, noisy: np.ndarray, sample_rate: int = 8000) -> np.ndarray:
    """
    Compute "oracle" optimal alpha values given clean reference.
    
    Strategy:
    - High SNR frames (speech heavy): alpha = 0.92 (responsive)
    - Medium SNR: alpha = 0.96 (balanced)
    - Low SNR (noise heavy): alpha = 0.99 (smooth)
    
    Returns:
        alpha: (n_frames,) values in [0.9, 0.99]
    """
    enhancer = LogMMSEEnhancer(sample_rate=sample_rate)
    
    # Get spectra
    clean_spec = enhancer.stft(clean)
    noisy_spec = enhancer.stft(noisy)
    
    clean_power = np.abs(clean_spec) ** 2
    noisy_power = np.abs(noisy_spec) ** 2
    
    n_frames = min(clean_power.shape[0], noisy_power.shape[0])
    
    # Compute frame-level SNR
    clean_power = clean_power[:n_frames]
    noisy_power = noisy_power[:n_frames]
    noise_power = np.abs(noisy_spec[:n_frames] - clean_spec[:n_frames]) ** 2
    
    frame_snr = 10 * np.log10(
        (np.mean(clean_power, axis=1) + 1e-10) / 
        (np.mean(noise_power, axis=1) + 1e-10)
    )
    
    # Map SNR to alpha
    # SNR > 10dB: alpha = 0.92 (responsive)
    # SNR 0-10dB: alpha = 0.92-0.98 (linear)
    # SNR < 0dB: alpha = 0.98-0.99 (smooth)
    
    alpha = np.zeros(n_frames)
    
    # High SNR (speech)
    high_snr_mask = frame_snr > 10
    alpha[high_snr_mask] = 0.92
    
    # Medium SNR
    medium_snr_mask = (frame_snr >= 0) & (frame_snr <= 10)
    alpha[medium_snr_mask] = 0.92 + (10 - frame_snr[medium_snr_mask]) * 0.006
    
    # Low SNR (noise)
    low_snr_mask = frame_snr < 0
    alpha[low_snr_mask] = 0.98 + np.clip(-frame_snr[low_snr_mask] / 20, 0, 0.01)
    
    return np.clip(alpha, 0.9, 0.99)


# ============================================
# Dataset
# ============================================

class AlphaDataset(Dataset):
    """Dataset for training alpha predictor."""
    
    def __init__(self, features_list: List[np.ndarray], alpha_list: List[np.ndarray], seq_len: int = 20):
        """
        Args:
            features_list: List of (n_frames, 5) feature arrays
            alpha_list: List of (n_frames,) alpha arrays
            seq_len: Sequence length for GRU
        """
        self.seq_len = seq_len
        self.samples = []
        
        for features, alpha in zip(features_list, alpha_list):
            n_frames = min(len(features), len(alpha))
            
            # Create overlapping sequences
            for i in range(0, n_frames - seq_len, seq_len // 2):
                feat_seq = features[i:i+seq_len]
                alpha_seq = alpha[i:i+seq_len]
                self.samples.append((feat_seq, alpha_seq))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        feat, alpha = self.samples[idx]
        return torch.FloatTensor(feat), torch.FloatTensor(alpha)


def create_dataset_from_noizeus(noizeus_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create training data from NOIZEUS dataset.
    
    Returns:
        features_list, alpha_list
    """
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile required")
    
    noizeus_path = Path(noizeus_dir)
    clean_dir = noizeus_path / "clean"
    
    if not clean_dir.exists():
        # Try alternative structure
        clean_dirs = list(noizeus_path.glob("*clean*"))
        if clean_dirs:
            clean_dir = clean_dirs[0]
        else:
            raise FileNotFoundError(f"Clean directory not found in {noizeus_dir}")
    
    features_list = []
    alpha_list = []
    
    # Find noisy directories
    noisy_dirs = [d for d in noizeus_path.iterdir() 
                  if d.is_dir() and "clean" not in d.name.lower() and d.suffix != '.zip']
    
    print(f"Processing {len(noisy_dirs)} noise conditions...")
    
    for noisy_dir in noisy_dirs:
        # Check for subdirectory (NOIZEUS structure)
        subdirs = [d for d in noisy_dir.iterdir() if d.is_dir()]
        if subdirs:
            noisy_dir = subdirs[0]
        
        wav_files = sorted(noisy_dir.glob("*.wav"))
        
        for wav_file in wav_files:
            # Find matching clean file
            # NOIZEUS naming: sp01_babble_sn10.wav -> sp01.wav
            clean_name = wav_file.stem.split("_")[0] + ".wav"
            clean_path = clean_dir / clean_name
            
            if not clean_path.exists():
                continue
            
            try:
                noisy, sr = sf.read(wav_file)
                clean, _ = sf.read(clean_path)
                
                # Ensure same length
                min_len = min(len(noisy), len(clean))
                noisy = noisy[:min_len]
                clean = clean[:min_len]
                
                # Extract features and oracle alpha
                features = extract_features(noisy, sample_rate=sr)
                alpha = compute_oracle_alpha(clean, noisy, sample_rate=sr)
                
                # Ensure same length
                min_frames = min(len(features), len(alpha))
                features = features[:min_frames]
                alpha = alpha[:min_frames]
                
                features_list.append(features)
                alpha_list.append(alpha)
                
            except Exception as e:
                print(f"  Error processing {wav_file.name}: {e}")
                continue
    
    print(f"Created {len(features_list)} training samples")
    return features_list, alpha_list


# ============================================
# Model
# ============================================

class TinyGRU(nn.Module):
    """
    Tiny GRU for predicting optimal alpha values.
    
    Total parameters: ~5K
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 32):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 5) features
            
        Returns:
            alpha: (batch, seq_len) in [0.9, 0.99]
        """
        # GRU
        out, _ = self.gru(x)  # (batch, seq_len, hidden)
        
        # FC + sigmoid to [0, 1], then scale to [0.9, 0.99]
        alpha = torch.sigmoid(self.fc(out))  # (batch, seq_len, 1)
        alpha = 0.9 + 0.09 * alpha.squeeze(-1)  # (batch, seq_len)
        
        return alpha
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# Training
# ============================================

def train_model(
    model: TinyGRU,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = "cpu"
) -> List[float]:
    """Train the model."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for features, alpha_target in train_loader:
            features = features.to(device)
            alpha_target = alpha_target.to(device)
            
            optimizer.zero_grad()
            alpha_pred = model(features)
            loss = criterion(alpha_pred, alpha_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses


def save_model(model: TinyGRU, path: str):
    """Save model to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 5,
        'hidden_size': 32,
    }, path)
    print(f"Model saved to {path}")


def load_model(path: str) -> TinyGRU:
    """Load model from file."""
    checkpoint = torch.load(path, map_location='cpu')
    model = TinyGRU(
        input_size=checkpoint.get('input_size', 5),
        hidden_size=checkpoint.get('hidden_size', 32)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


# ============================================
# Main Training Script
# ============================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Tiny GRU for adaptive alpha')
    parser.add_argument('--noizeus', type=str, default='noizeus_dataset', 
                        help='Path to NOIZEUS dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, default='tiny_gru_alpha.pth', 
                        help='Output model path')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Training Tiny GRU for Adaptive Alpha")
    print("="*60)
    
    # Create dataset
    print("\n[1/4] Creating training data from NOIZEUS...")
    features_list, alpha_list = create_dataset_from_noizeus(args.noizeus)
    
    if not features_list:
        print("Error: No training data created")
        return
    
    dataset = AlphaDataset(features_list, alpha_list, seq_len=20)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"  Training samples: {len(dataset)}")
    
    # Create model
    print("\n[2/4] Creating model...")
    model = TinyGRU(input_size=5, hidden_size=32)
    print(f"  Parameters: {model.count_parameters()}")
    
    # Train
    print(f"\n[3/4] Training for {args.epochs} epochs...")
    losses = train_model(model, train_loader, epochs=args.epochs)
    
    # Save
    print("\n[4/4] Saving model...")
    save_model(model, args.output)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Model saved: {args.output}")
    print(f"Model size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
