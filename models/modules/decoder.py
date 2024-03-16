import torch.nn as nn

from models.utils.init import initialize_weights


class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, patch_size: int):
        assert patch_size in [11, 27]
        super().__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.Upsample(size=(3, 3)),  # 1*1 ==> 3*3
            nn.BatchNorm2d(256),
            self.relu
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=(2, 2)),  # 3*3 ==> 6*6
            nn.BatchNorm2d(128),
            self.relu
        )
        if patch_size == 11:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.Upsample(size=(11, 11)),  # 6*6 ==> 11*11
                nn.BatchNorm2d(64),
                self.relu,
                nn.Conv2d(64, out_channels, 3, 1, 1)
            )
        elif patch_size == 27:
            self.layer3 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),  # 6*6 ==> 12*12
                nn.Upsample(scale_factor=(2, 2)),
                nn.BatchNorm2d(128),
                self.relu,

                nn.Conv2d(128, 64, 3, 1, 1),
                nn.Upsample(size=(27, 27)),  # 12*12 ==> 27*27

                nn.Conv2d(64, out_channels, 3, 1, 1)
            )
        else:
            raise NotImplementedError("Not support patch size {}".format(patch_size))
        self.model = nn.Sequential(self.layer1, self.layer2, self.layer3)
        initialize_weights(self.model)

    def forward(self, x):
        features = self.model(x)
        return features
