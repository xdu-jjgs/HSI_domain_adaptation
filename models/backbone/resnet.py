import torch.nn as nn
import torchvision.models as models

from models.utils.init import initialize_weights
from models.utils.download import load_pretrained_models


class ResNet(nn.Module):
    def __init__(self, in_channels: int, depth: int, pretrained=True,
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
        self.in_channels = in_channels
        self.out_channels = depth2channels[depth]

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
        model.conv1 = nn.Conv2d(self.in_channels, model.conv1.out_channels, 7, stride=2, padding=3, bias=False)
        if not pretrained:
            initialize_weights(model)

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
