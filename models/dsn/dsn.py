import torch
import torch.nn as nn
from typing import List

from models.modules import Gate, Decoder
from models.backbone import ImageClassifier
from models.utils.init import initialize_weights
from models.utils.hook import register_layer_hook
from tllib.modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer


class DSN(nn.Module):
    # TODO: denoise autoencoder
    def __init__(self, num_classes: int, experts: List[nn.Module], patch_size: int):
        super(DSN, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_source_encoder, self.shared_encoder, self.private_target_encoder = experts
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        if task_ind == 1:
            private_features = self.private_source_encoder(x)
        else:
            private_features = self.private_target_encoder(x)
        decoder_output = self.shared_decoder(torch.add(shared_features, private_features))
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output, decoder_output


class DSN_NoDis(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module], patch_size: int):
        super(DSN_NoDis, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_source_encoder, self.shared_encoder, self.private_target_encoder = experts
        self.shared_decoder = Decoder(self.out_channels, self.num_channels, patch_size)
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)
        register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        if task_ind == 1:
            private_features = self.private_source_encoder(x)
        else:
            private_features = self.private_target_encoder(x)
        decoder_output = self.shared_decoder(torch.add(shared_features, private_features))
        class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output, decoder_output


class DSN_Gate(DSN):
    def __init__(self, num_classes: int, experts: List[nn.Module], patch_size: int):
        super(DSN_Gate, self).__init__(num_classes, experts, patch_size)
        self.num_task = 2
        self.gates = nn.ModuleList([Gate(self.num_channels, 2) for _ in range(self.num_task)])

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        if task_ind == 1:
            task_weight = self.gates[0](x)[-1].softmax(dim=1).unsqueeze(1)
            private_features = self.private_source_encoder(x)
        else:
            task_weight = self.gates[1](x)[-1].softmax(dim=1).unsqueeze(1)
            private_features = self.private_target_encoder(x)
        experts_features = torch.stack([shared_features, private_features], 1)
        experts_features = torch.squeeze(experts_features)
        features_fusion = torch.matmul(task_weight, experts_features)
        features_fusion = features_fusion.view(features_fusion.size()[0], self.out_channels, 1, 1)
        decoder_output = self.shared_decoder(features_fusion)
        # TODO: 完整特征≠判别性特征？
        class_output = self.classifier(shared_features)[-1]
        reverse_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_features)[-1]
        return shared_features, private_features, class_output, domain_output, decoder_output, task_weight


class DSN_NoDecoder(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DSN_NoDecoder, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_source_encoder, self.shared_encoder, self.private_target_encoder = experts
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        if task_ind == 1:
            private_features = self.private_source_encoder(x)
        else:
            private_features = self.private_target_encoder(x)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output


class DSN_NoDecoder_NoDis(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DSN_NoDecoder_NoDis, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_source_encoder, self.shared_encoder, self.private_target_encoder = experts
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        initialize_weights(self)
        register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        if task_ind == 1:
            private_features = self.private_source_encoder(x)
        else:
            private_features = self.private_target_encoder(x)
        class_output = self.classifier(shared_features)[-1]
        return shared_features, private_features, class_output


class DSN_NoDecoder_NoSpec(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DSN_NoDecoder_NoSpec, self).__init__()
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_encoder, self.shared_encoder = experts
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)
        initialize_weights(self)
        register_layer_hook(self)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        shared_features = self.shared_encoder(x)
        private_features = self.private_encoder(x)
        class_output = self.classifier(shared_features)[-1]
        reverse_shared_features = self.grl(shared_features)
        domain_output = self.domain_discriminator(reverse_shared_features)[-1]
        return shared_features, private_features, class_output, domain_output
