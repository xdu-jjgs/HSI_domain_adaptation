import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels: int):
        super(PositionAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels, (1, 1), stride=1, padding=0)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        B = self.conv1(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # N*C(N=H*W)
        C = self.conv2(x).view(batch_size, -1, height * width)  # C*N
        D = self.conv3(x).view(batch_size, channels, height * width)  # C*N

        S = F.softmax(torch.bmm(B, C), dim=-1)  # N*C X C*N -> N*N
        E = torch.bmm(D, S.permute(0, 2, 1))  # C*N X N*N -> C*N
        E = E.view(batch_size, channels, height, width)  # C*N -> C*H*W
        E = x + self.alpha * E
        return E


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        B = x.view(batch_size, channels, height * width).permute(0, 2, 1)  # N*C
        C = x.view(batch_size, channels, height * width)  # C*N
        D = x.view(batch_size, channels, height * width)  # C*N

        X = F.softmax(torch.bmm(C, B), dim=-1)  # C*N X N*C -> C*C
        X = torch.max(X, -1, keepdim=True)[0].expand_as(X) - X
        # permute?
        E = torch.bmm(X.permute(0, 2, 1), D)  # C*C X C*N -> C*N
        E = E.view(batch_size, channels, height, width)
        E = x + self.beta * E
        return E
