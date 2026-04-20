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

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes),
        )

    def _format_input(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError("The input tensor dimension must be 3 (B, C, L) or (B, L, C)")
        b, a, c = x.shape
        if a == 2:
            x = x.transpose(1, 2)
        elif c == 2:
            pass
        else:
            raise ValueError("The last or second-to-last dimension must be the number of channels, which is 2")
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
