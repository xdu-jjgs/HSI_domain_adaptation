import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction='mean')

    @property
    def num_keys(self):
        return 4

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        print(torch.numel(images), torch.numel(reconstructions))

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return margin_loss + 0.5*reconstruction_loss


class MarginLoss(nn.Module):
    def __init__(self):
        super(MarginLoss, self).__init__()

    @property
    def num_keys(self):
        return 2

    def forward(self, classes, labels):
        '''
        :param labels: one-hot label please attention [batchsize, num_classes]
        :param classes: pred by CapsNet [batchsise, num_classes]
        :return: Variable contains a scalar loss value.
        '''
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum(dim=1).mean()

        return margin_loss
