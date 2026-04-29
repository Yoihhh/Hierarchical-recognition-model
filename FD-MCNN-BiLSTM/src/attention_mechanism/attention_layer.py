"""
Weak Signal Attention Module
This module is designed to enhance weak signal representations in mixed-signal scenarios.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SEBlock(nn.Module):
    # Squeeze-and-Excitation (SE) Block for channel attention.
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        
        # Global average pooling (squeeze)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Channel-wise excitation (lightweight MLP via 1x1 conv)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class TimesFormerBlock(nn.Module):
    """
    Lightweight TimesFormer-style block.
    """
    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 4,
        mlp_hidden: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Linear projection for embedding alignment
        self.patch_embed = nn.Linear(embed_dim, embed_dim)

        # Temporal attention (along sequence dimension)
        self.time_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Spatial attention (across feature dimension)
        self.space_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.patch_embed(x)
        x2, _ = self.time_attn(x1, x1, x1)
        x3 = x1 + x2
        x4, _ = self.space_attn(x3, x3, x3)
        x5 = x3 + x4
        x5 = self.norm2(self.norm1(x5))
        x6 = self.mlp(x5)
        out = self.norm3(x5 + x6)
        return out


class WeakSignalAttention(nn.Module):
    # Weak Signal Attention Module
    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 4,
        mlp_hidden: int = 4096,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Channel attention (enhance weak signal features)
        self.se = SEBlock(channels=2, reduction=se_reduction)

        # Lazy projection to embedding dimension
        self.input_proj = nn.LazyLinear(embed_dim)

        # Transformer-based attention
        self.timesformer = TimesFormerBlock(
            embed_dim=embed_dim, num_heads=num_heads, mlp_hidden=mlp_hidden
        )

    def _reshape_input(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            b, d = x.shape
            if d % 2 != 0:
                raise ValueError("Flat feature dim must be divisible by 2.")
            x = x.view(b, 2, d // 2)
        elif x.dim() == 3:
            if x.size(1) != 2:
                raise ValueError("Expected channel dimension = 2 for attention input")
        else:
            raise ValueError("Input must be 2D or 3D tensor.")
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._reshape_input(x)
        x_se = self.se(x)
        x_proj = self.input_proj(x_se)
        out = self.timesformer(x_proj)
        return out


__all__ = ["WeakSignalAttention"]
