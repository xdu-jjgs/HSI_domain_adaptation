import torch
import torch.nn as nn

from models.utils.init import initialize_weights
from models.backbone.extractor import FeatureExtractor


class Gate(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: bool = False):
        super(Gate, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            self.relu,
        )
        self.head = nn.Sequential(
            nn.Linear(64, num_classes)
        )
        if dropout:
            self.dropout = nn.Dropout(0.5)
            self.layer1.append(self.dropout)

        initialize_weights(self.layer1)
        initialize_weights(self.head)

    def forward(self, x):
        x = self.gap(x)
        while len(x.size()) > 2:
            x = torch.squeeze(x, 2)
        x = self.layer1(x)
        out = self.head(x)
        return x, out


class GateConv(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: bool = False):
        super(GateConv, self).__init__()
        self.extractor = FeatureExtractor(in_channels)
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channels, 128),
            self.relu,
        )
        self.head = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        if dropout:
            self.dropout = nn.Dropout(0.5)
            self.layer1.append(self.dropout)

        initialize_weights(self.layer1)
        initialize_weights(self.head)

    def forward(self, x):
        features = self.extractor(x)
        features = torch.squeeze(features)
        features = self.layer1(features)
        out = self.head(features)
        return features, out
