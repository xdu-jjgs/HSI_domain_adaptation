import torch
import torch.nn as nn
from typing import List

from models.backbone import ImageClassifier
from models.utils.init import initialize_weights
from models.utils.grad_filter import cal_grad_scores, mask_channels
from tllib.modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer


class S4DL_TS(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List):
        super(S4DL_TS, self).__init__()
        if hyperparams:
            assert 0. <= hyperparams[0] <= 1.
            self.filter_ratio = hyperparams[0]
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.backbone = backbone
        self.num_channels = backbone.in_channels
        self.out_channels = backbone.out_channels
        self.shared_encoder = nn.Conv2d(backbone.out_channels, self.out_channels, 3, 1, 1)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        shared_features = self.shared_encoder(features)
        private_features = features - shared_features
        return shared_features, private_features


class S4DL_DANN(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, hyperparams: List = None,
                 alpha: float = 1.0, lo: float = 1.0, hi: float = 1.0, max_iters: int = 1):
        super(S4DL_DANN, self).__init__()
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
        self.grl = WarmStartGradientReverseLayer(
            alpha=alpha, lo=lo, hi=hi, max_iters=max_iters, auto_step=True)
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        # register_layer_hook(self)

    def forward(self, x, task_ind, mask_di: bool = True, mask_ds: bool = True, return_features: bool = False):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        features = self.backbone(x)
        channels = x.size()[1]
        features_di = self.shared_encoder(features)
        features_ds = features - features_di
        reverse_features_di = self.grl(features_di)
        domain_output_di = self.domain_discriminator(reverse_features_di)[-1]
        if mask_di or mask_ds:
            assert 0. <= self.filter_ratio <= 1.
            filter_num = int(self.filter_ratio * channels)
            masked_features_di = features_di
            masked_features_ds = features_ds
            if mask_di:
                scores_di = cal_grad_scores(reverse_features_di, domain_output_di, task_ind)
                masked_features_di = mask_channels(features_di, scores_di, filter_num)
            if mask_ds:
                reverse_features_ds = self.grl(features_ds)
                domain_output_ds = self.domain_discriminator(reverse_features_ds)[-1]
                score_ds = cal_grad_scores(reverse_features_ds, domain_output_ds, task_ind)
                scores_abs = score_ds.abs()
                masked_features_ds = mask_channels(features_ds, scores_abs, filter_num, largest=False)
            class_output = self.classifier(masked_features_di)[-1]
            if return_features:
                return masked_features_di, masked_features_ds, class_output, domain_output_di, features_di, features_ds
            else:
                return masked_features_di, masked_features_ds, class_output, domain_output_di
        else:
            class_output = self.classifier(features_di)[-1]
        return features_di, features_ds, class_output, domain_output_di
