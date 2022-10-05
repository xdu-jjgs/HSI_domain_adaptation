# import torch.nn as nn
from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import *


class DDC(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(DDC, self).__init__()
        out_channels = 512
        self.feature_extractor = FeatureExtractor(in_channels, out_channels)
        self.classifier = ImageClassifier(out_channels, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        x, out = self.classifier(features)
        return x, out


class DDCondition(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, condition_l: int):
        super(DDCondition, self).__init__()
        out_channels = 512
        self.feature_extractor = FeatureExtractor(in_channels, out_channels)
        self.classifier = ImageClassifierCondition(out_channels, num_classes, condition_l)

    def forward(self, x, condition):
        features = self.feature_extractor(x)
        x, out = self.classifier(features, condition)
        return x, out
