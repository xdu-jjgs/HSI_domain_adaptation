from .ddc import DDC
from .dann import DANN
from configs import CFG
from models.backbone import build_backbone, ImageClassifier


def build_model(num_channels, num_classes):
    backbone_ = build_backbone(num_channels)
    if CFG.MODEL.NAME == 'ddc':
        return DDC(backbone_.out_channels, num_classes, backbone_)
    elif CFG.MODEL.NAME == 'mcd':
        FE = backbone_
        C1 = ImageClassifier(backbone_.out_channels, num_classes)
        C2 = ImageClassifier(backbone_.out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'dann':
        return DANN(backbone_.out_channels, num_classes, backbone_)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
