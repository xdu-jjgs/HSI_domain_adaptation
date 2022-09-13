import torch
import torch.nn as nn


class FocalLoss(nn.CrossEntropyLoss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_s, label_s):
        label_s = label_s.long()
        ce_loss = super(FocalLoss, self).forward(y_s, label_s)
        pt = torch.exp(-1 * ce_loss)
        # equals to:
        # pt = 1 / torch.exp(F.cross_entropy(input, target, reduction='none'))

        loss = self.alpha * torch.pow(1 - pt, self.gamma) * ce_loss
        # equals to
        # loss = -1 * self.alpha * torch.pow(1 - pt, self.gamma) * torch.log(pt)
        return loss
