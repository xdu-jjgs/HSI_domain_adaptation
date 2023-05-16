import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:   # TODO attention change here
            self.capsules = nn.ModuleList(
                [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(0, 0, 0))
                 for _ in range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet_3D(nn.Module):
    def __init__(self, num_channel, num_class):
        super(CapsuleNet_3D, self).__init__()

        # TODO: 改成3D卷积，每个3dconv中的outchannel=1，输出结果squeeze（dim=1）,由于49行的view函数，也可以不squeeze，最后的3d块数量由num_capsules决定相当于num_filters
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=256, kernel_size=3, stride=2)    # 27-->13
        self.primary_capsules = CapsuleLayer(num_capsules=32, num_route_nodes=-1, in_channels=1, out_channels=1,
                                             kernel_size=(7, 3, 3), stride=(4, 2, 2))    # 13-->6  channel: 256-->63
        self.digit_capsules = CapsuleLayer(num_capsules=num_class, num_route_nodes=63 * 6 * 6, in_channels=32,
                                           out_channels=16)

        # self.decoder = nn.Sequential(
        #     nn.Linear(16 * NUM_CLASSES, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 729),
        #     nn.Sigmoid()
        # )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = x.reshape(-1, 1, 256, 13, 13)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        # if y is None:
        #     # In all batches, get the most active capsule.
        #     _, max_length_indices = classes.max(dim=1)
        #     y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)
        #
        # reconstructions = self.decoder((x * y[:, :, None]).reshape(x.size(0), -1))

        # return classes, reconstructions
        return classes



