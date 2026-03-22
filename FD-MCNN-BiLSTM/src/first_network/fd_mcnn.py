"""FD-MCNN 第一层识别网络 / First-stage FD-MCNN detector.

输入：
    - iq: 形状 (B, 2, 2048) 的一维 IQ 数据（I/Q 两路）。
    - stft_map: 形状 (B, C, 384, 256) 的 STFT 时频图，可为单通道或三通道。

核心思路：
    1. 四个分支并行提取特征：
       - 时域分支：Conv1d 堆叠，侧重时序模式。
       - 调制特性分支：可分离卷积 + 通道注意力，突出调制相关特征。
       - 频域分支：Conv2d 堆叠，捕获时频局部模式。
       - 能量感知分支：Conv2d 堆叠，突出能量分布。
    2. 融合：
       - 时域输出与调制输出做通道拼接，再 1x1 卷积降维 + GAP 得到向量 v_tm。
       - 频域输出与能量输出做逐元素乘积，再 1x1 卷积降维 + GAP 得到向量 v_fe。
       - 拼接 v_tm 与 v_fe，送入 MLP 注意力加权得到 feat_for_attention。
    3. 分类头：线性映射得到 logits（Softmax 通常在损失外部计算）。

默认类别数 20，可根据数据集调整。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


def _same_padding_1d(kernel_size: int) -> int:
    """返回保持长度不变的 padding（stride=1）。"""
    return kernel_size // 2


def _same_padding_2d(ks: Tuple[int, int]) -> Tuple[int, int]:
    """返回保持高宽不变的 padding（stride=1）。"""
    kh, kw = ks
    return kh // 2, kw // 2


class SeparableConv1d(nn.Module):
    """一维深度可分离卷积：depthwise + pointwise。"""

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
    """简单通道注意力（SE 风格）。"""

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
        # x: (B, C, L)
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.mlp(w).view(b, c, 1)
        return x * w


class FDMcnnDetector(nn.Module):
    """FD-MCNN 第一层识别网络."""

    def __init__(self, num_classes: int = 20, reduce_channels: int = 64) -> None:
        """
        Args:
            num_classes: 输出类别数，默认 20。
            reduce_channels: 融合后降维的通道数 C。
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduce_channels = reduce_channels

        # 时域分支 (IQ)
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

        # 调制特性分支 (IQ)
        self.mod_branch = nn.Sequential(
            SeparableConv1d(2, 16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            SeparableConv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.mod_attn = ChannelAttention(32, reduction=8)

        # 频域分支 (STFT)
        self.freq_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), padding=_same_padding_2d((3, 1)), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 3), padding=_same_padding_2d((1, 3)), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 能量感知分支 (STFT)
        self.energy_branch = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=_same_padding_2d((5, 1)), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=_same_padding_2d((1, 5)), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 融合降维
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

        # 注意力加权 (向量级)
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * reduce_channels, reduce_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduce_channels, 2 * reduce_channels, bias=False),
            nn.Sigmoid(),
        )

        # 分类头
        self.classifier = nn.Linear(2 * reduce_channels, num_classes)

    def forward(self, iq: Tensor, stft_map: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            iq: (B, 2, 2048) 预处理后的一维 IQ。
            stft_map: (B, C, 384, 256) STFT 时频图，C 可为 1 或 3。

        Returns:
            logits: (B, num_classes)
            feat_for_attention: (B, 2 * reduce_channels)，供后续注意力层使用。
        """
        # 若 STFT 为多通道，先做通道平均为单通道
        if stft_map.size(1) != 1:
            stft_1ch = stft_map.mean(dim=1, keepdim=True)
        else:
            stft_1ch = stft_map

        # 时域 + 调制分支
        t_feat = self.time_branch(iq)          # (B,64,L)
        m_feat = self.mod_branch(iq)           # (B,32,L)
        m_feat = self.mod_attn(m_feat)
        tm_cat = torch.cat([t_feat, m_feat], dim=1)  # (B,96,L)
        tm_vec = self.reduce_time_mod(tm_cat).squeeze(-1)  # (B,C)

        # 频域 + 能量分支
        f_feat = self.freq_branch(stft_1ch)    # (B,32,H',W')
        e_feat = self.energy_branch(stft_1ch)  # (B,32,H',W')
        fe_mul = f_feat * e_feat               # 元素乘积
        fe_vec = self.reduce_freq_energy(fe_mul).squeeze(-1).squeeze(-1)  # (B,C)

        # 融合 + 注意力
        fused = torch.cat([tm_vec, fe_vec], dim=1)  # (B,2C)
        attn = self.attn_mlp(fused)
        feat_for_attention = fused * attn

        logits = self.classifier(feat_for_attention)
        return logits, feat_for_attention


__all__ = ["FDMcnnDetector"]


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FDMcnnDetector(num_classes=20, reduce_channels=64).to(device)
    model.eval()

    batch_size = 2
    iq = torch.randn(batch_size, 2, 2048, device=device)
    stft_map = torch.randn(batch_size, 1, 384, 256, device=device)

    with torch.no_grad():
        logits, feat_for_attention = model(iq, stft_map)

    print(f"device: {device}")
    print(f"iq shape: {tuple(iq.shape)}")
    print(f"stft_map shape: {tuple(stft_map.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"feat_for_attention shape: {tuple(feat_for_attention.shape)}")

    assert logits.shape == (batch_size, 20), "logits 输出形状不正确"
    assert feat_for_attention.shape == (batch_size, 128), "feat_for_attention 输出形状不正确"
    print("fd_mcnn.py 自测通过：前向输出正常。")
