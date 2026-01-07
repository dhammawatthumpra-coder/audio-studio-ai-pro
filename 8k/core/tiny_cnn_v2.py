"""
TinyCNN V2 - Improved 1D CNN for Alpha + SPP Prediction
========================================================
Based on TinyCNN from tiny_models.py with improvements:
- More layers for better feature extraction
- LayerNorm for stability
- Residual connections
- Same dual output as TinyGRU V2 (Alpha + SPP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class TinyCNNV2(nn.Module):
    """
    Improved 1D CNN for Alpha + SPP prediction.
    
    Architecture:
    - Input projection: 5 → hidden_size
    - 4 Conv layers with residual (kernel=5,5,3,3)
    - LayerNorm after each block
    - Dual heads: Alpha + SPP
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 16):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)
        
        # Conv blocks
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Dual prediction heads
        self.fc_alpha = nn.Conv1d(hidden_size, 1, 1)
        self.fc_spp = nn.Conv1d(hidden_size, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 5) features
            
        Returns:
            out: (batch, seq_len, 2) -> [Alpha, SPP]
        """
        # Transpose for Conv1d: (batch, seq, feat) -> (batch, feat, seq)
        x = x.transpose(1, 2)
        
        # Input projection
        h = self.input_proj(x)
        
        # Conv block 1 (residual)
        h1 = self.relu(self.conv1(h))
        h1 = self.dropout(h1)
        h1 = h1.transpose(1, 2)
        h1 = self.norm1(h1)
        h1 = h1.transpose(1, 2)
        h = h + h1
        
        # Conv block 2 (residual)
        h2 = self.relu(self.conv2(h))
        h2 = self.dropout(h2)
        h2 = h2.transpose(1, 2)
        h2 = self.norm2(h2)
        h2 = h2.transpose(1, 2)
        h = h + h2
        
        # Conv block 3 (residual)
        h3 = self.relu(self.conv3(h))
        h3 = h3.transpose(1, 2)
        h3 = self.norm3(h3)
        h3 = h3.transpose(1, 2)
        h = h + h3
        
        # Conv block 4 (residual)
        h4 = self.relu(self.conv4(h))
        h4 = h4.transpose(1, 2)
        h4 = self.norm4(h4)
        h4 = h4.transpose(1, 2)
        h = h + h4
        
        # Dual heads
        alpha = torch.sigmoid(self.fc_alpha(h))
        alpha = 0.9 + 0.09 * alpha
        
        spp = torch.sigmoid(self.fc_spp(h))
        
        # Transpose back
        alpha = alpha.transpose(1, 2)
        spp = spp.transpose(1, 2)
        
        return torch.cat([alpha, spp], dim=-1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Training utilities - optional import
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from tiny_gru_alpha import extract_features
    from tiny_gru_v2 import compute_oracle_targets_v2
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False


class DualTargetDataset(torch.utils.data.Dataset):
    """Dataset for Alpha + SPP targets."""
    
    def __init__(self, features_list, targets_list, seq_len=50):
        self.samples = []
        for features, targets in zip(features_list, targets_list):
            n_frames = min(len(features), len(targets))
            for start in range(0, n_frames - seq_len, seq_len // 2):
                end = start + seq_len
                f = features[start:end]
                t = targets[start:end]
                if len(f) == seq_len:
                    self.samples.append((
                        torch.FloatTensor(f),
                        torch.FloatTensor(t)
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def augment_snr(clean, noise, target_snr):
    """Augment by mixing at target SNR."""
    clean_power = np.mean(clean ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    snr_linear = 10 ** (target_snr / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    return clean + noise * scale


def train_cnn_v2(
    noizeus_dir: str, 
    epochs: int = 100, 
    output_path: str = "tiny_cnn_v2.pth",
    hidden_size: int = 16,
    augment: bool = True
):
    """Train TinyCNN V2 on NOIZEUS dataset."""
    import soundfile as sf
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("Training TinyCNN V2 (Alpha + SPP)")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/4] Loading NOIZEUS dataset...")
    noizeus_path = Path(noizeus_dir)
    clean_dir = list(noizeus_path.glob("*clean*"))[0]
    
    features_list = []
    targets_list = []
    
    noisy_dirs = [d for d in noizeus_path.iterdir() 
                  if d.is_dir() and "clean" not in d.name.lower() and d.suffix != '.zip']
    
    for noisy_dir in noisy_dirs:
        subdirs = [d for d in noisy_dir.iterdir() if d.is_dir()]
        if subdirs:
            noisy_dir = subdirs[0]
        
        for wav_file in sorted(noisy_dir.glob("*.wav")):
            clean_name = wav_file.stem.split("_")[0] + ".wav"
            clean_path = clean_dir / clean_name
            
            if not clean_path.exists():
                continue
            
            try:
                noisy, sr = sf.read(wav_file)
                clean, _ = sf.read(clean_path)
                
                min_len = min(len(noisy), len(clean))
                noisy = noisy[:min_len]
                clean = clean[:min_len]
                
                features = extract_features(noisy, sample_rate=sr)
                targets = compute_oracle_targets_v2(clean, noisy, sr)
                
                min_frames = min(len(features), len(targets))
                features_list.append(features[:min_frames])
                targets_list.append(targets[:min_frames])
                
                if augment:
                    noise_est = noisy - clean
                    for snr in [-5, 0, 5, 10, 15, 20]:
                        aug_noisy = augment_snr(clean, noise_est, snr)
                        aug_features = extract_features(aug_noisy, sr)
                        aug_targets = compute_oracle_targets_v2(clean, aug_noisy, sr)
                        n = min(len(aug_features), len(aug_targets))
                        features_list.append(aug_features[:n])
                        targets_list.append(aug_targets[:n])
                
            except Exception as e:
                continue
    
    print(f"  Loaded {len(features_list)} pairs")
    
    # 2. Create dataset
    dataset = DualTargetDataset(features_list, targets_list, seq_len=50)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"  Training samples: {len(dataset)}")
    
    # 3. Create model
    print("\n[2/4] Creating TinyCNN V2 model...")
    model = TinyCNNV2(input_size=5, hidden_size=hidden_size)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")
    print(f"  Hidden size: {hidden_size}")
    
    # 4. Train
    print(f"\n[3/4] Training for {epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            
            out = model(features)
            alpha_pred = out[:, :, 0]
            spp_pred = out[:, :, 1]
            
            alpha_target = targets[:, :, 0]
            spp_target = targets[:, :, 1]
            
            loss_alpha = F.mse_loss(alpha_pred, alpha_target)
            loss_spp = F.binary_cross_entropy(spp_pred, spp_target)
            loss = loss_alpha + 2.0 * loss_spp
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    # 5. Final save
    print(f"\n[4/4] Saving best model...")
    model.load_state_dict(torch.load(output_path, weights_only=True))
    torch.save(model.state_dict(), output_path)
    
    import os
    print(f"  Saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"  Best Loss: {best_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TinyCNN V2")
    parser.add_argument("--noizeus", required=True, help="Path to NOIZEUS dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--output", default="tiny_cnn_v2.pth")
    args = parser.parse_args()
    
    train_cnn_v2(args.noizeus, args.epochs, args.output, args.hidden)
