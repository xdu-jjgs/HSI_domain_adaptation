import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Distance(nn.Module):
    def __init__(self):
        super(L1Distance, self).__init__()
        # self.eps = eps

    @property
    def num_keys(self):
        return 2

    def forward(self, p1, p2):
        # target: tensor(b, c, h, w); prediction: same as target
        if not p1.shape == p2.shape:
            raise ValueError('Input images must have the same dimensions.')
        loss = torch.mean(torch.abs(F.softmax(p1, dim=1) - F.softmax(p2, dim=1)))
        return loss


class ExpMinusMSE(nn.Module):
    def __init__(self):
        super(ExpMinusMSE, self).__init__()
        self.reduction = 'mean'
        # self.eps = eps

    def forward(self, prediction, target):
        # target: tensor(b, c, h, w); prediction: same as target
        if not target.shape == prediction.shape:
            raise ValueError('Input images must have the same dimensions.')
        assert target.ndim == 4 and target.shape[1] > 1, "image n_channels should be greater than 1"
        mse = F.mse_loss(prediction, target, reduction=self.reduction)
        loss = torch.exp(-mse)
        return loss


class ExpMSE(nn.Module):
    def __init__(self):
        super(ExpMSE, self).__init__()
        self.reduction = 'mean'
        # self.eps = eps

    def forward(self, prediction, target):
        # target: tensor(b, c, h, w); prediction: same as target
        if not target.shape == prediction.shape:
            raise ValueError('Input images must have the same dimensions.')
        assert target.ndim == 4 and target.shape[1] > 1, "image n_channels should be greater than 1"
        mse = F.mse_loss(prediction, target, reduction=self.reduction)
        loss = torch.exp(mse)
        return loss
