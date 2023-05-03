import torch.nn as nn

from models.utils.init import initialize_weights
from models.modules import PositionAttentionModule, ChannelAttentionModule


class AttentionFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int=512, attention:str=''):
        super(AttentionFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert attention in ['pos', 'can']

        self.relu = nn.ReLU()
        self.convrelubn1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, 3, 2, 1),  # 27*27==>14*14
            nn.BatchNorm2d(64),
            self.relu,
        )
        self.convrelubn2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # 7*7
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.convrelubn3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # 4*4
            nn.BatchNorm2d(256),
            self.relu,
        )
        self.convrelubn4 = nn.Sequential(
            nn.Conv2d(256, out_channels, 3, 2, 0),  # 1*1
            nn.BatchNorm2d(out_channels),
            self.relu
        )
        if attention == 'pos':
            self.attention_module = PositionAttentionModule(256)
        else:
            self.attention_module = ChannelAttentionModule()

        initialize_weights(self.convrelubn1)
        initialize_weights(self.convrelubn2)
        initialize_weights(self.convrelubn3)
        initialize_weights(self.convrelubn4)
        initialize_weights(self.attention_module)

    def forward(self, x):
        features = self.convrelubn1(x)
        features = self.convrelubn2(features)
        features = self.convrelubn3(features)
        features = self.attention_module(features)
        features = self.convrelubn4(features)
        return features
