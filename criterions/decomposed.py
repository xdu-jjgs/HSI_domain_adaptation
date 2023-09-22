import torch
import torch.nn as nn


class OrthogonalDecomposed(nn.Module):
    def __init__(self):
        super(OrthogonalDecomposed, self).__init__()

    def forward(self, a, b, **kwargs):
        a = a.squeeze()
        b = b.squeeze()
        M = (torch.norm(a, p=2, dim=1, keepdim=True) * torch.norm(b, p=2, dim=1, keepdim=True))
        loss = torch.abs(torch.sum(M))
        return loss

