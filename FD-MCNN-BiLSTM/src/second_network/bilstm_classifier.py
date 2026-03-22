"""BiLSTM 第二层识别网络 / Second-stage BiLSTM classifier.

输入：
    - 来自注意力层的特征，形状 (B, 2, 2048) 或 (B, 2048, 2)。
    - 默认视为通道在前 (B, 2, 2048)，会自动转置为 LSTM 期望的 (B, 2048, 2)。

网络结构（按顺序）：
    BiLSTM(hidden=128) -> BiLSTM(hidden=64) -> BiLSTM(hidden=16)
    -> 时间维全局平均池化 -> MLP: 32→64→16→num_classes
    输出 logits（Softmax 在损失或推理阶段调用）。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn, Tensor


class BiLstmClassifier(nn.Module):
    """多层双向 LSTM + MLP 的弱信号分类器。"""

    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        # 三层 BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # 输出 (B, 2048, 256)
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # 输出 (B, 2048, 128)
        self.lstm3 = nn.LSTM(
            input_size=128,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )  # 输出 (B, 2048, 32)

        # MLP 头
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def _format_input(self, x: Tensor) -> Tensor:
        """确保输入为 (B, 2048, 2)。"""
        if x.dim() != 3:
            raise ValueError("输入张量维度需为 3 (B, C, L) 或 (B, L, C)。")
        b, a, c = x.shape
        if a == 2:  # (B, 2, L)
            x = x.transpose(1, 2)  # -> (B, L, 2)
        elif c == 2:  # 已是 (B, L, 2)
            pass
        else:
            raise ValueError("最后或倒数第二个维度必须为通道数 2。")
        return x

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 2, 2048) 或 (B, 2048, 2)
        Returns:
            logits: (B, num_classes)
        """
        x = self._format_input(x)  # (B, L, 2)

        out, _ = self.lstm1(x)  # (B, L, 256)
        out, _ = self.lstm2(out)  # (B, L, 128)
        out, _ = self.lstm3(out)  # (B, L, 32)

        # 时间维全局平均池化
        out = out.mean(dim=1)  # (B, 32)

        logits = self.head(out)  # (B, num_classes)
        return logits


__all__ = ["BiLstmClassifier"]


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 2048
    num_classes = 20

    model = BiLstmClassifier(num_classes=num_classes)
    model.eval()

    x_bcl = torch.randn(batch_size, 2, seq_len)
    x_blc = torch.randn(batch_size, seq_len, 2)

    with torch.no_grad():
        logits_bcl = model(x_bcl)
        logits_blc = model(x_blc)

    print("Input  (B, 2, 2048) -> logits:", tuple(logits_bcl.shape))
    print("Input  (B, 2048, 2) -> logits:", tuple(logits_blc.shape))

    expected_shape = (batch_size, num_classes)
    assert tuple(logits_bcl.shape) == expected_shape
    assert tuple(logits_blc.shape) == expected_shape
    assert torch.isfinite(logits_bcl).all()
    assert torch.isfinite(logits_blc).all()

    print("BiLstmClassifier self-test passed.")
