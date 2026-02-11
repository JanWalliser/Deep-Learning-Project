from __future__ import annotations

import torch
import torch.nn as nn


class CNNNet(nn.Module):
    """Simple MNIST CNN baseline.

    This is intentionally small and conventional, so results are easy to interpret.

    Architecture:
      Conv(3x3) -> ReLU -> MaxPool(2)
      Conv(3x3) -> ReLU -> MaxPool(2)
      Flatten -> FC -> ReLU -> Dropout -> FC
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int] = (32, 64),
        fc_dim: int = 128,
        dropout: float = 0.25,
        num_classes: int = 10,
    ):
        super().__init__()
        c1, c2 = int(channels[0]), int(channels[1])

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
