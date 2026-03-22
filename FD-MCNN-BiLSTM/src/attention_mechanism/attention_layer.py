"""Channel-Temporal Attention Layer.

Input:
- Supports (B, 2, D) or flat (B, D_flat).
- Flat input will be reshaped to (B, 2, D_flat/2).

Output:
- Enhanced feature tensor with shape (B, 2, embed_dim).
- Default embed_dim is 2048.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel re-weighting."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, L)
        w = self.pool(x)  # (B, C, 1)
        w = self.fc(w)    # (B, C, 1)
        return x * w


class TimesFormerBlock(nn.Module):
    """Lightweight TimesFormer: Time Attention -> Space Attention -> MLP."""

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 4,
        mlp_hidden: int = 4096,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(embed_dim, embed_dim)
        self.time_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.space_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
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
    """Weak-signal attention layer: SEBlock + TimesFormer."""

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 4,
        mlp_hidden: int = 4096,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.se = SEBlock(channels=2, reduction=se_reduction)
        # Project input feature dim D to embed_dim (initialized on first pass).
        self.input_proj = nn.LazyLinear(embed_dim)
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
                raise ValueError("Expected channel dimension = 2 for attention input.")
        else:
            raise ValueError("Input must be 2D or 3D tensor.")
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Return tensor with shape (B, 2, embed_dim)."""
        x = self._reshape_input(x)    # (B, 2, D)
        x_se = self.se(x)             # (B, 2, D)
        x_proj = self.input_proj(x_se)  # (B, 2, embed_dim)
        out = self.timesformer(x_proj)  # (B, 2, embed_dim)
        return out


__all__ = ["WeakSignalAttention"]


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    input_dim = 128
    output_dim = 2048

    model = WeakSignalAttention(
        embed_dim=output_dim,
        num_heads=4,
        mlp_hidden=4096,
    )
    model.eval()

    with torch.no_grad():
        # Case 1: 3D input (B, 2, D)
        x_3d = torch.randn(batch_size, 2, input_dim)
        y_3d = model(x_3d)
        assert y_3d.shape == (batch_size, 2, output_dim), (
            f"3D input output shape mismatch: got {tuple(y_3d.shape)}"
        )
        print("[PASS] 3D input test")
        print(f"Input shape:  {tuple(x_3d.shape)}")
        print(f"Output shape: {tuple(y_3d.shape)}")
        print(
            f"Output stats: mean={y_3d.mean().item():.6f}, "
            f"std={y_3d.std().item():.6f}"
        )

        # Case 2: 2D input (B, D*2) -> reshape to (B, 2, D)
        x_2d = torch.randn(batch_size, input_dim * 2)
        y_2d = model(x_2d)
        assert y_2d.shape == (batch_size, 2, output_dim), (
            f"2D input output shape mismatch: got {tuple(y_2d.shape)}"
        )
        print("[PASS] 2D input test")
        print(f"Input shape:  {tuple(x_2d.shape)}")
        print(f"Output shape: {tuple(y_2d.shape)}")
        print(
            f"Output stats: mean={y_2d.mean().item():.6f}, "
            f"std={y_2d.std().item():.6f}"
        )

    print("WeakSignalAttention self-test finished successfully.")
