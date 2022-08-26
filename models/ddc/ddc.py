import torch
import torch.nn as nn

from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import ImageClassifier


class DDC(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super(DDC, self).__init__()
        out_channels = 512
        self.feature_extractor = FeatureExtractor(in_channels, out_channels)
        self.classifier = ImageClassifier(out_channels, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        xs, out = self.classifier(features)
        return xs, out
