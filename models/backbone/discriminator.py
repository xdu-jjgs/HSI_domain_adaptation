import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, in_channel=512, num_class=10, prob=0.5):
        super(Discriminator, self).__init__()
        self.in_nodes = in_channel
        self.fc1 = nn.Linear(in_channel, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Linear(100, num_class)
        # self.bn_fc3 = nn.BatchNorm1d(num_class)
        self.prob = prob

    def forward(self, x):
        x = x.reshape([-1, self.in_nodes])
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=self.prob)
        x = self.fc3(x)
        return x
