import torch.nn as nn

from tllib.modules.grl import WarmStartGradientReverseLayer

from models.backbone import ImageClassifier


class DST(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DST, self).__init__()
        self.backbone = backbone
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)
        self.classifier_adv = ImageClassifier(backbone.out_channels, num_classes)
        self.classifier_pse = ImageClassifier(backbone.out_channels, num_classes)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=True)

    def forward(self, x):
        features = self.backbone(x)
        _, out = self.classifier(features)
        _, out_pse = self.classifier_pse(features)
        features_ = self.grl_layer(features)
        _, out_adv = self.classifier_adv(features_)
        return features, out, out_pse, out_adv
