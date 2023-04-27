from .gate import Gate
from .resnet import ResNet
from .extractor import FeatureExtractor
from .classifier import ImageClassifier, MultiHeadClassifier

__all__ = [
    Gate,
    ResNet,
    FeatureExtractor,
    ImageClassifier,
    MultiHeadClassifier
]


def build_backbone(num_channels, model_name):
    if model_name == 'resnet18':
        return ResNet(num_channels, depth=18)
    elif model_name == 'resnet34':
        return ResNet(num_channels, depth=34)
    elif model_name == 'resnet50':
        return ResNet(num_channels, depth=50)
    elif model_name == 'resnet101':
        return ResNet(num_channels, depth=101)
    elif model_name == 'fe':
        return FeatureExtractor(num_channels)
    raise NotImplementedError('invalid model: {}'.format(model_name))
