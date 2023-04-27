import torch
import torch.nn as nn

from typing import List

from models.backbone import Gate, ImageClassifier


class TaskMMOEDDC(nn.Module):
    def __init__(self, num_task_classes: List[int], experts: List[nn.Module]):
        super(TaskMMOEDDC, self).__init__()
        # backbone输入通道数
        self.num_channels = experts[0].in_channels
        self.num_task = len(num_task_classes)
        self.num_task_classes = num_task_classes
        self.experts = nn.ModuleList(experts)
        self.gates = nn.ModuleList([Gate(self.num_channels, len(experts)) for _ in range(self.num_task)])
        self.towers = nn.ModuleList([ImageClassifier(experts[0].out_channels, num_classes)
                                     for num_classes in num_task_classes])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        experts_features = [i(x) for i in self.experts]
        experts_features = torch.stack(experts_features, 1)
        while len(experts_features.size()) > 3:
            experts_features = torch.squeeze(experts_features, 3)
        x_gap = self.gap(x)

        task_weights = [i(x_gap)[-1].softmax(dim=1).unsqueeze(1) for i in self.gates]
        outs = []
        for i in range(self.num_task):
            # print(task_weights[i].size(), experts_features.size())
            features = torch.matmul(task_weights[i], experts_features)
            features = features.squeeze(1)
            # print(features.size())
            outs.append(self.towers[i](features))
        outs.append(task_weights)
        return outs
