from configs import CFG
from .resnet import ResNet
from .extractor import FeatureExtractor
from .classifier import ImageClassifier

__all__ = [
    ResNet,
    FeatureExtractor,
    ImageClassifier,
]


def build_backbone(num_channels):
    if CFG.MODEL.BACKBONE == 'resnet18':
        return ResNet(num_channels, depth=18)
    elif CFG.MODEL.BACKBONE == 'resnet34':
        return ResNet(num_channels, depth=34)
    elif CFG.MODEL.BACKBONE == 'resnet50':
        return ResNet(num_channels, depth=50)
    elif CFG.MODEL.BACKBONE == 'resnet101':
        return ResNet(num_channels, depth=101)
    elif CFG.MODEL.BACKBONE == 'fe':
        return FeatureExtractor(num_channels)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
