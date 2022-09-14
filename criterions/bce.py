import torch.nn as nn


class BCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s, **kwargs):
        y_s = y_s.squeeze(1)  # (n, c, h, w) -> (n, h, w)
        label_s = label_s.float()
        return super(BCELoss, self).forward(y_s, label_s)


class SigmoidBCELoss(nn.BCEWithLogitsLoss):
    def __init__(self, **kwargs):
        super(SigmoidBCELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s, **kwargs):
        y_s = y_s.squeeze(1)
        label_s = label_s.float()
        return super(SigmoidBCELoss, self).forward(y_s, label_s)
