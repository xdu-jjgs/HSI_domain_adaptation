import torch.nn as nn

from models.utils.init import initialize_weights
from tllib.modules.grl import GradientReverseLayer
from models.backbone.classifier import ImageClassifier


class VDD(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(VDD, self).__init__()
        self.backbone = backbone
        self.relu = nn.LeakyReLU()
        self.out_channels = backbone.out_channels
        self.di_extractor = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            self.relu
        )
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(backbone.out_channels, 2)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1)) 1*1 no need for gap

        initialize_weights(self.di_extractor)
        initialize_weights(self.classifier)
        initialize_weights(self.domain_discriminator)

    def forward(self, x):
        features = self.backbone(x)
        di_features = self.di_extractor(features)
        class_output = self.classifier(features)[-1]

        ds_features = features - di_features
        reverse_features = self.grl(ds_features)
        domain_output = self.domain_discriminator(reverse_features)[-1]
        return di_features, ds_features, class_output, domain_output

    def freeze_backbone(self):
        self.backbone.eval()

    def activate_backbone(self):
        self.backbone.train()


class VDDFixed(VDD):
    def forward(self, x):
        features = self.backbone(x)
        di_features = self.di_extractor(features)
        class_output = self.classifier(features)[-1]

        ds_features = features - di_features
        reverse_features = self.grl(di_features)
        domain_output = self.domain_discriminator(reverse_features)[-1]
        return di_features, ds_features, class_output, domain_output
