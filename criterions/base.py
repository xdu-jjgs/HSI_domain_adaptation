import torch.nn as nn


class TransferLoss(nn.Module):
    def __init__(self):
        super(TransferLoss, self).__init__()
