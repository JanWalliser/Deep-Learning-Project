import torch
import torch.nn as nn


class CNNNet(nn.Module):
    def __init__(self, in_channels = 1, channels = (32,64), fc_dim = 128, dropout = 0.25, num_classes = 10):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(nn.Conv2d(in_channels, c1, kerne_size = 3, padding = 1),
                                      nn.ReLU(),
                                      nnMaxPool2d(2),
                                      nn.Conv2d(c1, c2, kernel_size = 3, padding = 1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Linear(c2 * 7 * 7, fc_dim),
                                         nn.ReLU(),
                                         nn.Dropout(dropout),
                                         nn.Linear(fc_dim, num_classes))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.classifier(x)
            return x