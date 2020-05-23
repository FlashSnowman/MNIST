import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, 28 * 28))
        return self.layer1(x)


class CNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = torch.reshape(self.layer1(x), (-1, 3 * 3 * 128))
        return self.layer2(x)
