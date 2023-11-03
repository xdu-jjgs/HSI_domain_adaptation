from configs import CFG

from .resnet import ResNet
from .revnet import RevNet
from .extractor import FeatureExtractor
from .extractor_attention import AttentionFeatureExtractor
from .classifier import ImageClassifier, MultiHeadClassifier

__all__ = [
    ResNet,
    RevNet,
    FeatureExtractor,
    ImageClassifier,
    MultiHeadClassifier,
    AttentionFeatureExtractor
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
    elif model_name == 'revnet38':
        return RevNet(num_channels, depth=38)
    elif model_name == 'revnet110':
        return RevNet(num_channels, depth=110)
    elif model_name == 'revnet164':
        return RevNet(num_channels, depth=164)
    elif model_name == 'fe':
        return FeatureExtractor(num_channels, patch_size=CFG.DATASET.PATCH.HEIGHT)
    elif model_name == 'fe_pos':
        return AttentionFeatureExtractor(num_channels, attention='pos')
    elif model_name == 'fe_can':
        return AttentionFeatureExtractor(num_channels, attention='can')
    raise NotImplementedError('invalid model: {}'.format(model_name))
