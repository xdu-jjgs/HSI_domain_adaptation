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
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1) for _ in
                 range(num_capsules)])

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


class CapsuleNet(nn.Module):
    def __init__(self, num_channel, num_class):
        super(CapsuleNet, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=256, kernel_size=3, stride=2)    # 27--》13
        self.model = nn.Sequential(
            nn.Conv2d(num_channel, 256, 3, 2, 1),  # 27*27==>14*14
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, 2, 1),  # 7*7
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # nn.Conv2d(128, 256, 3, 2, 1),  # 4*4
            # nn.BatchNorm2d(256),
            # nn.ReLU(),

            # nn.Conv2d(256, out_channels, 3, 2, 0),  # 1*1
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        )
        self.primary_capsules = CapsuleLayer(num_capsules=32, num_route_nodes=-1, in_channels=128, out_channels=64,
                                             kernel_size=3, stride=2)    # 7--》4
        self.digit_capsules = CapsuleLayer(num_capsules=num_class, num_route_nodes=64 * 4 * 4, in_channels=32,
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
        # x = F.relu(self.conv1(x), inplace=True)
        x = self.model(x)
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


