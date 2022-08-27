import torch.nn as nn

from typing import List

from criterions.mmd import BaseMMDLoss


class LossComposer(nn.Module):
    def __init__(self, items: List[nn.Module], weights: List[float]):
        super(LossComposer, self).__init__()
        self.items = items
        self.weights = weights
        assert len(self.items) == len(self.weights), 'The number of loss items {} does not match weights {}'.format(
            len(self.items), len(self.weights))

    def forward(self, input, target, **kwargs):
        res = 0.
        for weight, item in zip(self.weights, self.items):
            if isinstance(item, BaseMMDLoss):
                l = weight * item(**kwargs)
            else:
                l = weight * item(input, target)
            res += l
        return res
