import torch.nn as nn
from models.backbone.domain_bn import DomainBatchNorm2d
from models.utils.init import initialize_weights


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, patch_size: int = 11):
        super(FeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 512
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 2, 1),  # 27*27->13*13/ 11*11->6*6
            DomainBatchNorm2d(64),
            self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 6*6/3*3
            DomainBatchNorm2d(128),
            self.relu,
        )
        if patch_size == 11:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),  # 3*3
                DomainBatchNorm2d(256),
                self.relu,
            )
        elif patch_size == 27:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1),  # 3*3
                DomainBatchNorm2d(256),
                self.relu,
            )
        else:
            raise NotImplementedError("Not supported patch size {}".format(patch_size))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 3, 0),  # 1*1
            DomainBatchNorm2d(512),
            self.relu
        )
        self.model = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

        initialize_weights(self.model)

    def forward(self, x):
        features = self.model(x)
        return features
