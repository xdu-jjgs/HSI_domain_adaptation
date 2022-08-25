import torch.nn as nn


class CELoss(nn.NLLLoss):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        target = target.long()
        return super(CELoss, self).forward(input, target)


class SoftmaxCELoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super(SoftmaxCELoss, self).__init__(**kwargs)

    def forward(self, input, target):
        target = target.long()
        return super(SoftmaxCELoss, self).forward(input, target)
