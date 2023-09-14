import torch
import torch.nn as nn

from typing import List

from tllib.alignment.cdan import RandomizedMultiLinearMap
from tllib.modules.grl import WarmStartGradientReverseLayer, GradientReverseLayer

from models.backbone import ImageClassifier, MultiHeadClassifier
from models.modules import Gate, ReverseLayerF


class DDFEMMOEDANN(nn.Module):
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DDFEMMOEDANN, self).__init__()
        # backbone输入通道数
        self.num_channels = experts[0].in_channels
        self.num_task = 2  # source domain and target domain
        self.num_classes = num_classes
        # self.domain_invariant_expert, self.source_specific_expert, self.target_specific_expert = experts
        self.domain_invariant_expert, self.source_specific_expert,_ = experts
        self.gates_1 = nn.ModuleList([Gate(self.num_channels, 2) for _ in range(self.num_task)])
        # self.gates_2 = Gate(self.num_channels, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.grl_layer_1 = GradientReverseLayer()
        self.grl_layer_2 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.2, max_iters=100, auto_step=True)
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.classifier_adv = MultiHeadClassifier(experts[0].out_channels, [num_classes, 2])
        self.classifier_pse = ImageClassifier(experts[0].out_channels, num_classes)
        self.mapping = RandomizedMultiLinearMap(experts[0].out_channels, num_classes, experts[0].out_channels)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2, 3]
        x_gap = self.gap(x)
        experts_features = [self.domain_invariant_expert(x)]
        if task_ind < 3:
            source_specific_features = self.source_specific_expert(x)
            if task_ind == 1:
                experts_features.append(source_specific_features)
                task_weight = self.gates_1[0](x_gap)[-1].softmax(dim=1).unsqueeze(1)
            else:
                experts_features.append(self.grl_layer_1(source_specific_features))
                task_weight = self.gates_1[1](x_gap)[-1].softmax(dim=1).unsqueeze(1)
        else:
            target_specific_features = self.target_specific_expert(x)
            experts_features.append(target_specific_features)
            task_weight = self.gates_2(x_gap)[-1].softmax(dim=1).unsqueeze(1)
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)

        if task_ind == 1:
            _, out = self.classifier(features)
            return out, task_weight
        elif task_ind == 2:
            _, out = self.classifier(features)
            _, out_pse = self.classifier_pse(features)
            features_ = self.grl_layer_2(features)
            out_ = out.detach()
            features_mapping = self.mapping(features_, out_)
            _, outs_adv = self.classifier_adv(features_mapping)
            out_adv_k, out_adv_d = outs_adv
            return out, out_pse, out_adv_k, out_adv_d, task_weight
        elif task_ind == 3:
            _, out_pse = self.classifier_pse(features)
            return out_pse, task_weight

    def freeze_domain_invariant(self):
        self.domain_invariant_expert.eval()
        self.classifier_pse.eval()

    def train_domain_invariant(self):
        self.domain_invariant_expert.train()
        self.classifier_pse.train()
