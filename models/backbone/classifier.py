import torch
import torch.nn as nn

from typing import List

from models.utils.init import initialize_weights


class ImageClassifier(nn.Module):
    def __init__(self, in_nodes: int, num_classes: int, dropout: bool = False):
        super(ImageClassifier, self).__init__()
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_nodes, 256),
            self.relu,
        )
        # change 64 to 100 if load failed
        self.layer2 = nn.Sequential(
            nn.Linear(256, 100),
            self.relu
        )
        self.head = nn.Sequential(
            nn.Linear(100, num_classes)
        )
        if dropout:
            self.dropout = nn.Dropout(0.5)
            self.layer1.append(self.dropout)
            self.layer2.append(self.dropout)

        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        initialize_weights(self.head)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.head(x)
        return x, out


class MultiHeadClassifier(nn.Module):
    def __init__(self, in_nodes: int, heads: List[int], dropout: bool = False):
        super(MultiHeadClassifier, self).__init__()
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_nodes, 256),
            self.relu,
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 100),
            self.relu
        )
        self.heads = nn.ModuleList()
        for h in heads:
            self.heads.append(nn.Sequential(nn.Linear(100, h)))
        if dropout:
            self.dropout = nn.Dropout(0.5)
            self.layer1.append(self.dropout)
            self.layer2.append(self.dropout)

        initialize_weights(self.layer1)
        initialize_weights(self.layer2)
        initialize_weights(self.heads)

    def forward(self, x):
        x = torch.squeeze(x)
        x = self.layer1(x)
        x = self.layer2(x)
        outs = []
        for h in self.heads:
            outs.append(h(x))
        return x, outs


class SingleLayerClassifier(nn.Module):
    def __init__(self, in_nodes: int, num_classes: int, drop_ratio: float = 0.):
        super(SingleLayerClassifier, self).__init__()
        self.drop_ratio = drop_ratio
        self.relu = nn.LeakyReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(in_nodes, num_classes)
        )
        initialize_weights(self.head)

    def cal_scores(self, features, labels):
        features_ = features.detach()
        weights = self.head.weight.clone().detach()  # num_domains x C
        print(weights.size(), features_.size(), labels.size())
        domain_num, channel_num = weights.shape[0], weights.shape[1]
        weights = weights[labels]
        batch_size, _ = features_.shape # features after gap
        scores = torch.mul(weights, features_)
        return scores

    def forward(self, x):
        x = self.gap(x)
        x = torch.squeeze(x)
        out = self.head(x)

        return x, out
