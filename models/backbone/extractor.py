import torch.nn as nn
from models.utils.init import initialize_weights


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 512):
        super(FeatureExtractor, self).__init__()
        self.out_channels = out_channels

        self.relu = nn.ReLU()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),  # 27*27==>14*14
            nn.BatchNorm2d(64),
            self.relu,

            nn.Conv2d(64, 128, 3, 2, 1),  # 7*7
            nn.BatchNorm2d(128),
            self.relu,

            nn.Conv2d(128, 256, 3, 2, 1),  # 4*4
            nn.BatchNorm2d(256),
            self.relu,

            nn.Conv2d(256, out_channels, 3, 2, 0),  # 1*1
            nn.BatchNorm2d(out_channels),
            self.relu
        )

        initialize_weights(self.model)

    def forward(self, x):
        features = self.model(x)
        return features
