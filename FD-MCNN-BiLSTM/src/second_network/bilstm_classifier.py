from __future__ import annotations

import torch
from torch import nn, Tensor


class BiLstmClassifier(nn.Module):
    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm3 = nn.LSTM(
            input_size=128,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # MLP 头
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def _format_input(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("输入张量维度需为 3 (B, C, L) 或 (B, L, C)。")
        b, a, c = x.shape
        if a == 2:
            x = x.transpose(1, 2)
        elif c == 2:
            pass
        else:
            raise ValueError("最后或倒数第二个维度必须为通道数 2。")
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._format_input(x)

        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)

        out = out.mean(dim=1)

        logits = self.head(out)
        return logits


__all__ = ["BiLstmClassifier"]