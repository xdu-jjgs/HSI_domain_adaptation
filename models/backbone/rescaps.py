import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import torchvision.models as models

# from models.utils.download import load_pretrained_models


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
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
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

        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=512, kernel_size=1, stride=1)    # 1--》1
        self.primary_capsules = CapsuleLayer(num_capsules=32, num_route_nodes=-1, in_channels=512, out_channels=64,
                                             kernel_size=1, stride=1)    # 1--》1
        self.digit_capsules = CapsuleLayer(num_capsules=num_class, num_route_nodes=64 * 1 * 1, in_channels=32,
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


class ResCaps(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=False,
                 replace_stride_with_dilation=None):
        super(ResCaps, self).__init__()
        self.model_name = 'resnet{}'.format(depth)
        model = getattr(models, self.model_name)(replace_stride_with_dilation=replace_stride_with_dilation)
        depth2channels = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
        }
        out_channels = depth2channels[depth]

        # if pretrained:
        #     model = load_pretrained_models(model, self.model_name)
        model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, 7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.caps = CapsuleNet(num_channel=out_channels, num_class=num_classes)
        # self.avgpool = model.avgpool
        # self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        # print(1, x.shape)
        x = self.layer1(x)
        # print(2, x.shape)
        x = self.layer2(x)
        # print(3, x.shape)
        x = self.layer3(x)
        # print(4, x.shape)
        x = self.layer4(x)
        # print(5, x.shape)
        x = self.caps(x)
        # print(6, x.shape)
        # x = self.avgpool(x)
        # print(6, x.shape)
        # x = torch.flatten(x, 1)
        # print(7, x.shape)
        # x = self.fc(x)
        # print(8, x.shape)
        return x