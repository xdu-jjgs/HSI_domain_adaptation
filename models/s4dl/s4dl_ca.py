import torch
import torch.nn as nn
from typing import List

from models.backbone import ImageClassifier
from models.utils.init import initialize_weights
from tllib.modules.grl import GradientReverseLayer
from models.modules.attention import ChannelAttentionModule


class S4DL_CA(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List):
        super(S4DL_CA, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.channel_attn = ChannelAttentionModule()
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = features - shared_features
        shared_features = self.channel_attn(shared_features)
        private_features = self.channel_attn(private_features)
        class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output


class S4DL_CA_DANN(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List = None):
        super(S4DL_CA_DANN, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        self.channel_attn = ChannelAttentionModule()
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        channels = x.size()[1]
        features_di = self.shared_encoder(features)
        features_ds = features - features_di
        reverse_features_di = self.grl(features_di)
        domain_output_di = self.domain_discriminator(reverse_features_di)[-1]
        features_di = self.channel_attn(features_di)
        features_ds = self.channel_attn(features_ds)
        class_output = self.classifier(features_di)[-1]
        return features_di, features_ds, class_output, domain_output_di
