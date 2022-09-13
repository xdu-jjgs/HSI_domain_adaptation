import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, y_s, label_s):
        n = y_s.shape[0]

        y_s = y_s.view(n, -1)
        label_s = label_s.view(n, -1)
        intersection = y_s * label_s

        loss = 1 - (2 * (intersection.sum(1) + self.smooth) / (y_s.sum(1) + label_s.sum(1) + self.smooth)).sum() / n
        return loss


class SigmoidDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SigmoidDiceLoss, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, y_s, label_s):
        y_s = self.sigmoid(y_s)
        loss = self.dice(y_s, label_s)
        return loss
