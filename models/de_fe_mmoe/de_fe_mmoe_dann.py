import torch
import torch.nn as nn

from typing import List
from tllib.alignment.cdan import RandomizedMultiLinearMap
from tllib.modules.grl import WarmStartGradientReverseLayer, GradientReverseLayer

from models.modules import Gate
from models.backbone import ImageClassifier, MultiHeadClassifier


class DEFEMMOEDANN(nn.Module):
    # TODO: 1.update backbone
    #  2.add mapping
    #  3.try orthogonal loss
    #  4.try two steps
    #  5.update extractor of MMOE
    #  6.try different experts
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DEFEMMOEDANN, self).__init__()
        # backbone
        self.num_channels = experts[0].in_channels
        self.num_task = 2  # domain specific and domain invariant
        self.num_classes = num_classes
        self.experts = nn.ModuleList(experts)
        self.fft_modules = nn.ModuleList([
            nn.Conv2d(self.num_channels, self.num_channels, 1, 1, 0, bias=False)
            for _ in range(len(self.experts))
        ])
        self.gates = nn.ModuleList([Gate(self.num_channels, len(experts)) for _ in range(self.num_task)])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=200, auto_step=True)
        self.grl_layer = GradientReverseLayer()
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.domain_discriminator = ImageClassifier(experts[0].out_channels, 2)
        # self.mapping = RandomizedMultiLinearMap(experts[0].out_channels, num_classes, experts[0].out_channels)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        experts_features = []
        amplitude_features = []
        for fft_conv, expert in zip(self.fft_modules, self.experts):
            freq_tensor = torch.fft.fft2(x)
            amplitude = torch.abs(freq_tensor)
            phase = torch.angle(freq_tensor)
            amplitude_extract = fft_conv(amplitude)
            amplitude_features.append(amplitude_extract)
            complex_tensor = amplitude_extract * (torch.cos(phase) + 1j * torch.sin(phase))
            recon = torch.fft.ifft2(complex_tensor)
            recon = recon.real
            experts_features.append(expert(recon))
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)

        x_gap = self.gap(x)
        if task_ind == 1:
            task_weight = self.gates[0](x_gap)[-1].softmax(dim=1).unsqueeze(1)
        else:
            task_weight = self.gates[1](x_gap)[-1].softmax(dim=1).unsqueeze(1)
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)
        _, class_output = self.classifier(features)
        reverse_features = self.grl_layer(features)
        _, domain_output = self.domain_discriminator(reverse_features)
        return amplitude_features, class_output, domain_output, task_weight
