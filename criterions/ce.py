import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.NLLLoss):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s, **kwargs):
        label_s = label_s.long()
        return super(CELoss, self).forward(y_s, label_s)


class SoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(SoftmaxCELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s, **kwargs):
        label_s = label_s.long()
        return super(SoftmaxCELoss, self).forward(y_s, label_s)


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, y_s, label_s, mask, **kwargs):
        loss = F.cross_entropy(y_s, label_s, reduction='none') - F.kl_div(y_s, label_s, reduction='none')
        loss = (loss * mask).mean()
        return loss
