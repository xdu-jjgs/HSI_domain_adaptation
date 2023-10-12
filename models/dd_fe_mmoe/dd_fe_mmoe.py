import torch
import torch.nn as nn

from typing import List
from tllib.alignment.cdan import RandomizedMultiLinearMap
from tllib.modules.grl import WarmStartGradientReverseLayer, GradientReverseLayer

from models.modules import Gate
from models.backbone import ImageClassifier, MultiHeadClassifier


class DDFEMMOE(nn.Module):
    # TODO: 1. update backbone 2. add mapping 3.try orthogonal loss 4.try two steps
    def __init__(self, num_classes: int, experts: List[nn.Module]):
        super(DDFEMMOE, self).__init__()
        # backbone
        self.num_channels = experts[0].in_channels
        self.num_task = 2  # domain specific and domain invariant
        self.num_classes = num_classes
        self.domain_invariant_expert, self.source_specific_expert, self.target_specific_expert = experts
        # self.domain_invariant_expert, self.source_specific_expert, _ = experts
        self.gates = nn.ModuleList([Gate(self.num_channels, 2) for _ in range(self.num_task)])
        # self.gates_2 = Gate(self.num_channels, 2)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=1.0, max_iters=200, auto_step=True)
        self.classifier = ImageClassifier(experts[0].out_channels, num_classes)
        self.classifier_adv = MultiHeadClassifier(experts[0].out_channels, [num_classes, 2])
        self.classifier_pse = ImageClassifier(experts[0].out_channels, num_classes)
        # self.mapping = RandomizedMultiLinearMap(experts[0].out_channels, num_classes, experts[0].out_channels)

    def forward(self, x, task_ind):
        assert task_ind in [1, 2]  # 1 for source domain, 2 for target domain
        domain_invariant_features = self.domain_invariant_expert(x)
        experts_features = [domain_invariant_features]
        if task_ind == 1:
            domain_specific_features = self.source_specific_expert(x)
            task_weight = self.gates[0](x)[-1].softmax(dim=1).unsqueeze(1)
        else:
            domain_specific_features = self.target_specific_expert(x)
            task_weight = self.gates[1](x)[-1].softmax(dim=1).unsqueeze(1)
        experts_features.append(domain_specific_features)

        experts_features = torch.stack(experts_features, 1)
        experts_features = torch.squeeze(experts_features)
        features = torch.matmul(task_weight, experts_features)
        features = features.squeeze(1)

        _, out = self.classifier(features)  # C1
        _, out_pse = self.classifier_pse(features)  # C2
        features_ = self.grl_layer(features)
        # out_ = out.detach()
        # features_mapping = self.mapping(features_, out_)
        # _, outs_adv = self.classifier_adv(features_mapping)
        _, outs_adv = self.classifier_adv(features_)  # C3
        out_adv_k, out_adv_d = outs_adv
        return out, out_pse, out_adv_k, out_adv_d, task_weight

    def freeze_domain_invariant(self):
        self.domain_invariant_expert.eval()
        self.classifier_pse.eval()

    def train_domain_invariant(self):
        self.domain_invariant_expert.train()
        self.classifier_pse.train()
