import torch.nn as nn
from models.utils.init import initialize_weights


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super(FeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 2, 1),  # 27*27->13*13/ 11*11->6*6
            nn.BatchNorm2d(64),
            self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 6*6/3*3
            nn.BatchNorm2d(128),
            self.relu,
        )
        if self.out_channels > 256:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),  # 3*3
                nn.BatchNorm2d(256),
                self.relu,
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(256, out_channels, 3, 3, 0),  # 1*1
                nn.BatchNorm2d(out_channels),
                self.relu
            )
            self.model = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)
        else:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, out_channels, 3, 3, 1),  # 1*1
                nn.BatchNorm2d(256),
                self.relu,
            )
            self.model = nn.Sequential(self.layer1, self.layer2, self.layer3)

        initialize_weights(self.model)

    def forward(self, x):
        features = self.model(x)
        return features
