import torch
import torch.nn as nn
from typing import List
from tllib.modules.grl import GradientReverseLayer, WarmStartGradientReverseLayer

from models.backbone import ImageClassifier


class DSN(nn.Module):
    # TODO: denoise autoencoder
    def __init__(self, num_classes: int, experts: List[nn.Module], patch_size: int):
        super(DSN, self).__init__()
        assert patch_size in [11]
        # backbone输入通道数
        self.relu = nn.LeakyReLU()
        self.num_classes = num_classes
        self.num_channels = experts[0].in_channels
        self.out_channels = experts[0].out_channels
        self.private_source_encoder, self.shared_encoder, self.private_target_encoder = experts
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, 3, 1, 1),
            nn.Upsample(size=(3, 3)),  # 1*1 ==> 3*3
            nn.BatchNorm2d(256),
            self.relu,

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Upsample(scale_factor=(2, 2)),  # 3*3 ==> 6*6
            nn.BatchNorm2d(128),
            self.relu,

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Upsample(size=(11, 11)),  # 6*6 ==> 12*12
            nn.BatchNorm2d(64),
            self.relu,

            nn.Conv2d(64, self.num_channels, 3, 1, 1)
            # nn.Upsample(size=(23, 23)) # 12*12 ==> 23*23
        )
        self.classifier = ImageClassifier(self.out_channels, num_classes)
        self.grl = GradientReverseLayer()
        self.domain_discriminator = ImageClassifier(self.out_channels, 2)

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
