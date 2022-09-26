import torch
import torch.nn as nn
import torchvision.models as models

from models.utils.download import load_pretrained_models


class ResNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=True,
                 replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        self.model_name = 'resnet{}'.format(depth)
        model = getattr(models, self.model_name)(replace_stride_with_dilation=replace_stride_with_dilation)
        depth2channels = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
        }
        out_channels = depth2channels[depth]

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
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
        self.avgpool = model.avgpool
        self.fc = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNetCondition(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, depth: int, pretrained=True,
                 replace_stride_with_dilation=None, label_embedding=False):
        super(ResNet, self).__init__()
        self.model_name = 'resnet{}'.format(depth)
        self.label_embedding = label_embedding
        model = getattr(models, self.model_name)(replace_stride_with_dilation=replace_stride_with_dilation)
        depth2channels = {
            18: 512,
            34: 512,
            50: 2048,
            101: 2048,
        }
        out_channels = depth2channels[depth]

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
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
        self.avgpool = model.avgpool
        self.fc = nn.Linear(out_channels + num_classes, num_classes)
        if self.label_embedding:
            self.embedding = nn.Embedding(num_classes, 64)
            self.fc = nn.Linear(out_channels + 64, num_classes)

    def forward(self, x, label):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.label_embedding:
            label = self.embedding(label)
        x = torch.cat((x, label), dim=1)
        x = self.fc(x)
        return x
