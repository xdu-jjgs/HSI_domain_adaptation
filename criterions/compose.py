import torch.nn as nn

from typing import List

from criterions.mmd import TransferLoss


class LossComposer(nn.Module):
    def __init__(self, items: List[nn.Module], weights: List[float]):
        super(LossComposer, self).__init__()
        self.items = items
        self.weights = weights
        assert len(self.items) == len(self.weights), 'The number of loss items {} does not match weights {}'.format(
            len(self.items), len(self.weights))

    def forward(self, *args):
        res = 0.
        arg_index = 0
        for weight, item in zip(self.weights, self.items):
            key_num = item.num_keys
            item_args = args[arg_index:arg_index+key_num]
            if key_num == 2:
                l = weight * item(item_args[0], item_args[1])
            elif key_num == 3:
                l = weight * item(item_args[0], item_args[1], item_args[2])
            elif key_num == 4:
                l = weight * item(item_args[0], item_args[1], item_args[2], item_args[3])
            res += l
            arg_index += key_num
        return res

    # def forward(self, input, target, **kwargs):
    #     res = 0.
    #     for weight, item in zip(self.weights, self.items):
    #         if isinstance(item, TransferLoss):
    #             l = weight * item(**kwargs)
    #         else:
    #             l = weight * item(input, target)
    #         res += l
    #     return res


