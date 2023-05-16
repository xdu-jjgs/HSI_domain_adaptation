import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from models.backbone.cbam import CBAM
# from models.backbone.capsnet import CapsuleNet
# from models.backbone.rescaps import ResCaps


# class TwoBranchCaps(nn.Module):
#     def __init__(self, num_channels: int, num_classes: int):
#         super(TwoBranchCaps, self).__init__()
#         self.caps_1d = ResCaps(num_channels, num_classes, depth=18, pretrained=False)
#         self.caps_2d = CapsuleNet(num_channels, num_classes)
#
#     def forward(self, x):
#         classes = self.caps_1d(x) + self.caps_2d(x)
#         classes = nn.functional.softmax(classes, dim=-1)
#         return classes


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer2d(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer2d, self).__init__()

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
            # outputs = self.squash(outputs)

        return outputs


class CapsuleLayer1d(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=3):
        super(CapsuleLayer1d, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            # self.capsules = nn.ModuleList(
            #     [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
            #      range(num_capsules)])
            self.capsules = nn.ModuleList(
                [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1) for _ in
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
            # outputs = self.squash(outputs)

        return outputs


class TwoBranchCaps(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super(TwoBranchCaps, self).__init__()
        self.attention = CBAM(num_channels)
        self.fenet2d = nn.Sequential(
            nn.Conv2d(num_channels, 256, 3, 2, 1),  # 27*27==>14*14
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, 3, 2, 1),  # 7*7
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fenet1d = nn.Sequential(

            nn.Conv2d(128, 256, 3, 2, 1),  # 4*4
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, 2, 0),  # 1*1
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.primary_capsules2d = CapsuleLayer2d(num_capsules=32, num_route_nodes=-1, in_channels=128, out_channels=64,
                                               kernel_size=3, stride=2)  # 7--》4
        # self.digit_capsules2d = CapsuleLayer2d(num_capsules=num_classes, num_route_nodes=64 * 4 * 4, in_channels=32,
        #                                      out_channels=16)
        self.primary_capsules1d = CapsuleLayer1d(num_capsules=32, num_route_nodes=-1, in_channels=1, out_channels=1,
                                               kernel_size=3, stride=2)  # 1--》1  512-->256
        self.digit_capsules1d = CapsuleLayer1d(num_capsules=num_classes, num_route_nodes=256 + 64 * 4 * 4, in_channels=32,
                                             out_channels=16)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        x = self.attention(x)
        x2d = self.fenet2d(x)
        x1d = self.fenet1d(x2d)
        x1d = x1d.reshape(-1, 1, 512)  # when conv is 1d

        caps2d = self.primary_capsules2d(x2d)
        # print(caps2d.shape)
        caps1d = self.primary_capsules1d(x1d)
        # print(caps1d.shape)
        outputs = torch.cat((caps2d, caps1d), dim=1)
        # print(outputs.shape)
        outputs = self.squash(outputs)
        outputs = self.digit_capsules1d(outputs).squeeze().transpose(0, 1)

        classes = (outputs ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        return classes
