import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightCE(nn.Module):
    def __init__(self):
        super(ClassWeightCE, self).__init__()
        self.reduction = 'mean'

    @property
    def num_keys(self):
        return 4

    def forward(self, class_g, class_t, d_pred, d_target):
        d_target = d_target.long()
        class_g = F.softmax(class_g, dim=1)
        class_t = F.softmax(class_t, dim=1)
        d_pred = F.softmax(d_pred, dim=1)
        w = F.cross_entropy(class_g, class_t, reduction=self.reduction)
        w = 1.0 + torch.exp(-w)
        ce = F.cross_entropy(d_pred, d_target, reduction=self.reduction)
        loss = w * ce
        return loss
