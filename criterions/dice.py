import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, input, target):
        n = input.shape[0]

        input = input.view(n, -1)
        target = target.view(n, -1)
        intersection = input * target

        loss = 1 - (2 * (intersection.sum(1) + self.smooth) / (input.sum(1) + target.sum(1) + self.smooth)).sum() / n
        return loss


class SigmoidDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SigmoidDiceLoss, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, input, target):
        input = self.sigmoid(input)
        loss = self.dice(input, target)
        return loss
