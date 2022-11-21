import torch
import torch.nn as nn

from models.utils.init import initialize_weights


class ImageClassifier(nn.Module):
    def __init__(self, in_nodes: int, num_classes: int, dropout: bool = False):
        super(ImageClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_nodes, 256),
            self.relu,
        )
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
        while len(x.size()) > 2:
            x = torch.squeeze(x, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.head(x)
        return x, out
