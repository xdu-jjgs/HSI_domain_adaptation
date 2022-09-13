import torch.nn as nn


class CELoss(nn.NLLLoss):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s):
        label_s = label_s.long()
        return super(CELoss, self).forward(y_s, label_s)


class SoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(SoftmaxCELoss, self).__init__(**kwargs)

    def forward(self, y_s, label_s):
        label_s = label_s.long()
        return super(SoftmaxCELoss, self).forward(y_s, label_s)
