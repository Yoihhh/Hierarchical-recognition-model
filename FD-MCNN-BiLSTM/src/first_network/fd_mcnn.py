"""
FD-MCNN Detector
This module implements a multi-branch feature extraction network for signal classification.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


def _same_padding_1d(kernel_size: int) -> int:
    # Compute padding size to keep 1D feature length unchanged.
    return kernel_size // 2


def _same_padding_2d(ks: Tuple[int, int]) -> Tuple[int, int]:
    kh, kw = ks
    return kh // 2, kw // 2


class SeparableConv1d(nn.Module):
    # Depthwise separable 1D convolution.
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int) -> None:
        super().__init__()
        pad = _same_padding_1d(kernel_size)
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel_size, padding=pad, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelAttention(nn.Module):
    # Channel Attention (Squeeze-and-Excitation style).
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.mlp(w).view(b, c, 1)
        return x * w


class FDMcnnDetector(nn.Module):
    """
    FD-MCNN
    Designed for mixed signal recognition by combining:
        - Time-domain features
        - Modulation characteristics
        - Frequency-domain features
        - Energy distribution
    Final output is obtained via attention-weighted feature fusion.
    """
    def __init__(self, num_classes: int = 20, reduce_channels: int = 64) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reduce_channels = reduce_channels

        # Time-Domain Branch (IQ)
        self.time_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=7, padding=_same_padding_1d(7), bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, padding=_same_padding_1d(5), bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=_same_padding_1d(3), bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Modulation-Characteristics Branch (IQ)
        self.mod_branch = nn.Sequential(
            SeparableConv1d(2, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            SeparableConv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.mod_attn = ChannelAttention(32, reduction=8)

        # Frequency-Domain Branch (STFT)
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=_same_padding_2d((3, 1)), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=_same_padding_2d((1, 3)), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Energy-Sensing Branch (STFT)
        self.energy_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=_same_padding_2d((5, 1)), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=_same_padding_2d((1, 5)), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fusion Dimensionality Reduction
        self.reduce_time_mod = nn.Sequential(
            nn.Conv1d(96, reduce_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.reduce_freq_energy = nn.Sequential(
            nn.Conv2d(32, reduce_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Attention weighting
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduce_channels, 2 * reduce_channels, bias=False),
            nn.Sigmoid(),
        )

        self.classifier = nn.Linear(2 * reduce_channels, num_classes)

    def forward(self, iq: Tensor, stft_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            iq: time-domain IQ signal [B, 2, T]
            stft_map: spectrogram [B, C, H, W]

        Returns:
            logits: classification output
            feat_for_attention: fused feature representation
        """
        if stft_map.size(1) != 1:
            stft_1ch = stft_map.mean(dim=1, keepdim=True)
        else:
            stft_1ch = stft_map

        t_feat = self.time_branch(iq)
        m_feat = self.mod_branch(iq)
        m_feat = self.mod_attn(m_feat)
        tm_cat = torch.cat([t_feat, m_feat], dim=1)
        tm_vec = self.reduce_time_mod(tm_cat).squeeze(-1)

        f_feat = self.freq_branch(stft_1ch)
        e_feat = self.energy_branch(stft_1ch)
        fe_mul = f_feat * e_feat
        fe_vec = self.reduce_freq_energy(fe_mul).squeeze(-1).squeeze(-1)

        fused = torch.cat([tm_vec, fe_vec], dim=1)
        attn = self.attn_mlp(fused)
        feat_for_attention = fused * attn

        logits = self.classifier(feat_for_attention)
        return logits, feat_for_attention


__all__ = ["FDMcnnDetector"]
