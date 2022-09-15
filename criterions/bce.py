import torch.nn as nn


class BCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        input = input.squeeze(1)  # (n, c, h, w) -> (n, h, w)
        target = target.float()
        return super(BCELoss, self).forward(input, target)


class SigmoidBCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super(SigmoidBCELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.float()
        return super(SigmoidBCELoss, self).forward(input, target)

