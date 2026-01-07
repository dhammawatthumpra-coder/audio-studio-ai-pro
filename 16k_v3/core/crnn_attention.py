"""
CRNN with GRU + Attention: Best Architecture for Audio
=======================================================
- CNN: Extract spectral features
- GRU: Capture temporal dynamics
- Attention: Focus on relevant frames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention for real-time processing.
    Each frame can only attend to itself and past frames.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, hidden)
        Returns:
            out: (batch, seq, hidden)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask: prevent attending to future
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        out = self.out_proj(out)
        
        return out


class CRNN_GRU_Attention(nn.Module):
    """
    CRNN with GRU + Causal Self-Attention
    
    Architecture:
    1. CNN: Extract spectral features
    2. GRU: Capture temporal dynamics
    3. Attention: Focus on relevant past frames
    4. Output heads: Alpha and SPP
    
    ~110K params
    """
    
    def __init__(
        self, 
        input_size: int = 5, 
        cnn_channels: int = 48, 
        gru_hidden: int = 64, 
        gru_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, 7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, 3, padding=1),
            nn.ReLU(),
        )
        self.cnn_norm = nn.LayerNorm(cnn_channels)
        
        # GRU for temporal dynamics
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0,
            bidirectional=False  # Causal for real-time
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)
        
        # Causal Self-Attention
        self.attention = CausalSelfAttention(gru_hidden, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(gru_hidden)
        
        # Output heads
        self.alpha_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        self.spp_head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Linear(gru_hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        """
        Args:
            x: Input features (batch, seq, features)
            hidden: Optional GRU hidden state for streaming
        
        Returns:
            out: (batch, seq, 2) - Alpha and SPP
            hidden: GRU hidden state for next chunk
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN: (batch, seq, feat) -> (batch, feat, seq) -> CNN -> (batch, channels, seq)
        x = x.transpose(1, 2)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq, channels)
        cnn_out = self.cnn_norm(cnn_out)
        
        # GRU: (batch, seq, channels) -> (batch, seq, gru_hidden)
        if hidden is not None:
            gru_out, hidden = self.gru(cnn_out, hidden)
        else:
            gru_out, hidden = self.gru(cnn_out)
        gru_out = self.gru_norm(gru_out)
        
        # Attention with residual connection
        attn_out = self.attention(gru_out)
        attn_out = self.attn_norm(gru_out + attn_out)  # Residual
        
        # Output heads
        alpha = self.alpha_head(attn_out)
        spp = self.spp_head(attn_out)
        
        # Scale alpha to [0.9, 0.99]
        alpha = 0.9 + alpha * 0.09
        
        out = torch.cat([alpha, spp], dim=-1)  # (batch, seq, 2)
        
        return out, attn_out, hidden  # Include latent for forensics
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CRNN_GRU_Attention(input_size=5, cnn_channels=48, gru_hidden=64, gru_layers=2, num_heads=4)
    print(f"CRNN_GRU_Attention Parameters: {model.count_parameters():,}")
    
    x = torch.randn(2, 100, 5)
    out, latent, hidden = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Hidden: {hidden.shape}")
    print(f"Alpha range: [{out[:,:,0].min():.3f}, {out[:,:,0].max():.3f}]")
    print(f"SPP range: [{out[:,:,1].min():.3f}, {out[:,:,1].max():.3f}]")
