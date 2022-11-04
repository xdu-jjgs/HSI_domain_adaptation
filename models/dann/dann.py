import torch.nn as nn
from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import ImageClassifier
from models.modules.reverselayer import ReverseLayerF


class DANN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, backbone: nn.Module):
        super(DANN, self).__init__()
        self.feature_extractor = backbone
        self.classifier = ImageClassifier(in_channels, num_classes)
        self.domain_discriminator = ImageClassifier(in_channels, 2)

    def forward(self, x, alpha):
        features = self.feature_extractor(x)
        reverse_features = features.reshape([-1, self.out_channels])
        reverse_features = ReverseLayerF.apply(reverse_features, alpha)
        class_output = self.classifier(features)[-1]
        domain_output = self.domain_discriminator(reverse_features)[-1]
        return class_output, domain_output
