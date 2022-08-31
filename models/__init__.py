from configs import CFG
from .ddc import DDC


def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'ddc':
        return DDC(num_channels, num_classes)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
