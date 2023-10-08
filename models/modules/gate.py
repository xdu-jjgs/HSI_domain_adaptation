import torch
import torch.nn as nn

from models.utils.init import initialize_weights


class Gate(nn.Module):
    def __init__(self, in_nodes: int, num_classes: int, dropout: bool = False):
        super(Gate, self).__init__()
        self.relu = nn.LeakyReLU()
        self.layer1 = nn.Sequential(
            nn.Linear(in_nodes, 64),
            self.relu,
        )
        self.head = nn.Sequential(
            nn.Linear(64, num_classes)
        )
        if dropout:
            self.dropout = nn.Dropout(0.5)
            self.layer1.append(self.dropout)

        initialize_weights(self.layer1)
        initialize_weights(self.head)

    def forward(self, x):
        while len(x.size()) > 2:
            x = torch.squeeze(x, 2)
        x = self.layer1(x)
        out = self.head(x)
        return x, out
