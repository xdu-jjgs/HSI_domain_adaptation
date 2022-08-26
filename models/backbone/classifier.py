import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self, in_nodes: int, num_classes: int, dropout: bool = True):
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.head(x)
        return x, out
