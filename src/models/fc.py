import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, hidden_sizes=(256, 128), dropout=0.1, num_classes=10):
        super().__init__()
        layers = []
        in_dim = 28 * 28
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
