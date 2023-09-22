import torch
import torch.nn as nn


class OrthogonalDecomposed(nn.Module):
    def __init__(self):
        super(OrthogonalDecomposed, self).__init__()

    def forward(self, a, b, **kwargs):
        a = a.squeeze()
        b = b.squeeze()
        a_norm = a / torch.norm(a, p=2, dim=1, keepdim=True)
        b_norm = b / torch.norm(b, p=2, dim=1, keepdim=True)
        M = a_norm * b_norm
        loss = torch.mean(torch.abs(torch.sum(M, dim=1)))
        return loss

