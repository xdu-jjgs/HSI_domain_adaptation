import torch.nn as nn
import pytorchcv.models.revnet as revnet

from models.utils.init import initialize_weights
from models.utils.download import load_pretrained_models


class RevNet(nn.Module):
    def __init__(self, in_channels: int, depth: int, pretrained=True,
                 replace_stride_with_dilation=None):
        super(RevNet, self).__init__()
        self.model_name = 'revnet{}'.format(depth)
        model: RevNet = getattr(revnet, self.model_name)()
        depth2channels = {
            38: 112,
            110: 128,
            164: 512,
        }
        self.in_channels = in_channels
        self.out_channels = depth2channels[depth]

        if pretrained:
            model = load_pretrained_models(model, self.model_name)
        model.init_block_conv = nn.Conv2d(self.in_channels, model.conv1.out_channels, (3, 3), stride=(1, 1),
                                          padding=(1, 1), bias=False)
        if not pretrained:
            initialize_weights(model)

        self.init_block = nn.Sequential(
            model.init_block_conv,
            model.features.init_block.bn,
            model.features.init_block.activ
        )
        self.stage1 = model.features.stage1
        self.stage2 = model.features.stage2
        self.stage3 = model.features.stage3
        self.final_postactiv = model.features.final_postactiv

    def forward(self, x):
        x = self.init_block(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_postactiv(x)
        return x
