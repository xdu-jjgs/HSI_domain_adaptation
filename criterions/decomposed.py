import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalDecomposed(nn.Module):
    def __init__(self):
        super(OrthogonalDecomposed, self).__init__()

    def forward(self, a, b, **kwargs):
        a = a.squeeze()
        b = b.squeeze()
        print(a.size(), b.size())
        M = (torch.norm(a, p=2, dim=1) ** 2 * torch.norm(b, p=2, dim=1) ** 2)
        loss = torch.mean(torch.abs(torch.sum(M, dim=1)))
        return loss
