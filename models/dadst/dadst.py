import torch
import torch.nn as nn

from tllib.alignment.cdan import RandomizedMultiLinearMap
from tllib.modules.grl import WarmStartGradientReverseLayer

from models.backbone import ImageClassifier, MultiHeadClassifier


class DADST(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DADST, self).__init__()
        self.backbone = backbone
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)
        self.classifier_adv = MultiHeadClassifier(backbone.out_channels, [num_classes, 2])
        self.classifier_pse = ImageClassifier(backbone.out_channels, num_classes)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=200, auto_step=True)

    def forward(self, x):
        features = self.backbone(x)
        _, out = self.classifier(features)
        _, out_pse = self.classifier_pse(features)
        features_ = self.grl_layer(features)
        _, outs_adv = self.classifier_adv(features_)
        out_adv_k, out_adv_d = outs_adv
        return features, out, out_pse, out_adv_k, out_adv_d


class DADASTMapping(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module):
        super(DADASTMapping, self).__init__()
        self.backbone = backbone
        self.classifier = ImageClassifier(backbone.out_channels, num_classes)
        self.classifier_adv = MultiHeadClassifier(backbone.out_channels, [num_classes, 2])
        self.classifier_pse = ImageClassifier(backbone.out_channels, num_classes)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=200, auto_step=True)
        self.mapping = RandomizedMultiLinearMap(backbone.out_channels, num_classes, backbone.out_channels)

    def forward(self, x):
        features = self.backbone(x)
        _, out = self.classifier(features)
        _, out_pse = self.classifier_pse(features)
        features_ = self.grl_layer(features)
        out_ = out.detach()
        # TODO: fix bug about O2 Half and Float
        features_ = torch.squeeze(features_)
        features_mapping = self.mapping(features_, out_)
        _, outs_adv = self.classifier_adv(features_mapping)
        out_adv_k, out_adv_d = outs_adv
        return features, out, out_pse, out_adv_k, out_adv_d