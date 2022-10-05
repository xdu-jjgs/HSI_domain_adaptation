import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidWeightCE(nn.Module):
    def __init__(self):
        super(ConfidWeightCE, self).__init__()
        self.reduction = 'mean'

    @property
    def num_keys(self):
        return 3

    def forward(self, pred_class, pred_domain, label_domain):
        pred_class = F.softmax(pred_class, dim=1)
        pred_domain = F.softmax(pred_domain, dim=1)
        label_domain = label_domain.long()
        w = -pred_class * (torch.log(pred_class + 1e-5))    # entropy
        w = torch.sum(w, dim=1)
        w = 1.0 + torch.exp(-w)
        w = torch.mean(w)
        ce = F.cross_entropy(pred_domain, label_domain, reduction=self.reduction)
        # loss = torch.mean(w * ce)
        loss = w * ce
        return loss


# pseudo_label = torch.rand(64, 7)
# domain = torch.rand(64, 2)
# label = torch.ones(64)
# loss = ConfidWeightCE().forward(pseudo_label, domain, label)
# print(loss)

