import torch
import torch.nn as nn

from typing import List

from models.backbone import Gate, ImageClassifier


class MMOE(nn.Module):
    def __init__(self, num_task_classes: List[int], experts: List[nn.Module]):
        super(MMOE, self).__init__()
        self.num_task = len(num_task_classes)
        self.num_task_classes = num_task_classes
        self.experts = experts
        self.gates = [Gate(1024, self.num_task) for _ in range(self.num_task)]
        self.towers = [ImageClassifier(experts[0].out_channels, num_classes) for num_classes in num_task_classes]

    def forward(self, x):
        experts_features = [i(x) for i in self.experts]
        experts_features = torch.stack(experts_features, 1)
        task_weights = [i(x)[-1].softmax(dim=1) for i in self.gates]
        outs = []
        for i in range(self.num_task):
            features = torch.matmul(task_weights[i], experts_features)
            outs.append(self.towers[i](features))
        return outs

