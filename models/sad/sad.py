import torch.nn as nn

from models.utils.init import initialize_weights
from tllib.modules.grl import GradientReverseLayer
from models.backbone.classifier import ImageClassifier


class SAD(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(SAD, self).__init__()
        self.backbone = backbone
        self.relu = nn.LeakyReLU()
        self.out_channels = backbone.out_channels
        self.di_extractor = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            self.relu
        )
        self.ds_extractor = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            self.relu
        )
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.sar = ImageClassifier(backbone.out_channels, 2)
        self.domain_discriminator = ImageClassifier(backbone.out_channels, 2)
        # self.gap = nn.AdaptiveAvgPool2d((1, 1)) 1*1 no need for gap

        initialize_weights(self.di_extractor)
        initialize_weights(self.ds_extractor)
        initialize_weights(self.classifier)
        initialize_weights(self.sar)
        initialize_weights(self.domain_discriminator)

    def forward(self, x):
        features = self.backbone(x)
        di_features = self.di_extractor(features)
        ds_features = self.ds_extractor(features)
        class_output = self.classifier(di_features)[-1]
        reverse_features = self.grl(di_features)
        domain_output = self.domain_discriminator(reverse_features)[-1]
        sar_output_di = self.sar(di_features)[-1]
        sar_output_ds = self.sar(ds_features)[-1]
        return di_features, class_output, domain_output, sar_output_di, sar_output_ds

