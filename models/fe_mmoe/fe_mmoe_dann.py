import torch
import torch.nn as nn

from typing import List
from tllib.modules.grl import WarmStartGradientReverseLayer

from models.backbone import ImageClassifier
from models.modules import Gate, ReverseLayerF


class FEMMOEDANN(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(FEMMOEDANN, self).__init__()
        # backbone输入通道数
        self.num_channels = experts[0].in_channels
        self.num_task = 2  # source domain and target domain
        self.num_classes = num_classes
        self.experts = nn.ModuleList(experts)
        self.gates = nn.ModuleList([Gate(self.num_channels, len(experts)) for _ in range(self.num_task)])
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.domain_discriminator = ImageClassifier(experts[0].out_channels, 2)
        # self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)

    def forward(self, x, alpha, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain and 2 for target domain
        experts_features = [i(x) for i in self.experts]
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)

        if task_ind == 1:
            task_weight = self.gates[0](x)[-1].softmax(dim=1).unsqueeze(1)
        else:
            task_weight = self.gates[1](x)[-1].softmax(dim=1).unsqueeze(1)
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)
        class_output = self.classifier(features)
        # reverse_features = self.grl_layer(features)
        reverse_features = features.reshape([-1, self.experts[0].out_channels])
        reverse_features = ReverseLayerF.apply(reverse_features, alpha)
        domain_output = self.domain_discriminator(reverse_features)
        return class_output, domain_output, task_weight
