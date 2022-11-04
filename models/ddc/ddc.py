import torch.nn as nn

from models.backbone import FeatureExtractor
from models.backbone import ImageClassifier


class DDC(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, backbone: nn.Module):
        super(DDC, self).__init__()
        self.feature_extractor = backbone
        self.classifier = ImageClassifier(in_channels, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        x, out = self.classifier(features)
        return x, out
