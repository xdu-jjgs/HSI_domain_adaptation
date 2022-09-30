import torch.nn as nn

from typing import List, Dict


class LossComposer(nn.Module):
    def __init__(self, items: List[nn.Module], weights: List[float]):
        super(LossComposer, self).__init__()
        self.items = items
        self.weights = weights
        assert len(self.items) == len(self.weights), 'The number of loss items {} does not match weights {}'.format(
            len(self.items), len(self.weights))

    def forward(self, **kwargs) -> Dict:
        res = {
            'total': 0.
        }
        for weight, item in zip(self.weights, self.items):
            l = weight * item(**kwargs)
            res[item.__class__.__name__] = l
            res['total'] += l
        return res
