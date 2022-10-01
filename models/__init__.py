from configs import CFG
from .ddc import DDC
from .dann import DANN
from models.backbone.extractor import FeatureExtractor
from models.backbone.classifier import ImageClassifier


def build_model(num_channels, num_classes):
    if CFG.MODEL.NAME == 'ddc':
        return DDC(num_channels, num_classes)
    elif CFG.MODEL.NAME == 'mcd':
        out_channels = 512
        FE = FeatureExtractor(num_channels, out_channels)
        C1 = ImageClassifier(out_channels, num_classes)
        C2 = ImageClassifier(out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'dann':
        return DANN(num_channels, num_classes)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
