import torch.nn as nn

from models.backbone import FeatureExtractor
from models.backbone import ImageClassifier


class DDC(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DDC, self).__init__()
        self.backbone = backbone
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        x, out = self.classifier(features)
        return features, x, out
