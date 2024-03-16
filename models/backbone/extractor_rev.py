import torch.nn as nn
from models.utils.init import initialize_weights
from pytorchcv.models.revnet import ReversibleBlock


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(BasicBlock, self).__init__()
        self.relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        fea = self.conv(x)
        fea = self.bn(fea)
        fea = self.relu(fea)
        return fea


class RevUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride: int, padding: int):
        super(RevUnit, self).__init__()
        assert in_channels % 2 == 0
        assert out_channels % 2 == 0
        in_channels2 = in_channels // 2
        out_channels2 = out_channels // 2
        self.resize_identity = (stride != 1)
        if self.resize_identity:
            self.adjust_block = BasicBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
            in_channels2 = out_channels2
            print(111111111111)
        gm = BasicBlock(in_channels2, out_channels2, 3, 1, 1)
        fm = BasicBlock(in_channels2, out_channels2, 3, 1, 1)
        self.body_block = ReversibleBlock(gm, fm)

    def forward(self, x):
        fea = x
        if self.resize_identity:
            fea = self.adjust_block(fea)
        fea = self.body_block(fea)
        return fea


class RevFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int):
        super(RevFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = 128
        self.out_channels = 128

        self.relu = nn.LeakyReLU()
        self.layer1 = RevUnit(self.in_channels, self.mid_channels, stride=2, padding=1)  # 27*27->14*14 / 11*11->6*6
        self.layer2 = RevUnit(self.mid_channels, self.out_channels, stride=1, padding=1)  # 27*27 / 6*6
        self.layer3 = RevUnit(self.out_channels, self.out_channels, stride=1, padding=1)  # 6*6
        self.layer4 = RevUnit(self.out_channels, self.out_channels, stride=1, padding=1)  # 6*6
        self.layer5 = BasicBlock(self.out_channels, self.out_channels, kernel_size=3, stride=3, padding=1)  # 3*3
        self.layer6 = BasicBlock(self.out_channels, self.out_channels, kernel_size=3, stride=3, padding=1)  # 1*1
        self.model = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6)

        initialize_weights(self.model)

    def forward(self, x):
        features = self.model(x)
        return features
