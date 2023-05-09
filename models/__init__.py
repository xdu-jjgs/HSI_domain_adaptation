from .ddc import DDC
from .dqn import DQN
from .dst import DST
from .dann import DANN
from .dstda import DSTDA, DSTDAMapping
from .task_mmoe import TaskMMOEDDC, TaskMMOEDANN
from .fe_mmoe import FEMMOEDDC, FEMMOEDANN, FEMMOEMCD, FEMMOESol1

from configs import CFG
from models.backbone import build_backbone, ImageClassifier


def build_model(num_channels, num_classes):
    if CFG.MODEL.BACKBONE:
        backbone_ = build_backbone(num_channels, CFG.MODEL.BACKBONE)
    elif CFG.MODEL.EXPERTS[0]:
        backbone_ = [build_backbone(num_channels, i) for i in CFG.MODEL.EXPERTS]
    else:
        raise NotImplementedError('invalid backbone: {} or experts: {}'.format(CFG.MODEL.BACKBONE, CFG.MODEL.EXPERTS))
    if CFG.MODEL.NAME == 'ddc':
        return DDC(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'mcd':
        FE = backbone_
        C1 = ImageClassifier(backbone_.out_channels, num_classes)
        C2 = ImageClassifier(backbone_.out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'dann':
        return DANN(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dst':
        return DST(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dstda':
        return DSTDA(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dstda_mapping':
        return DSTDAMapping(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'task_mmoe_ddc':
        return TaskMMOEDDC(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'task_mmoe_dann':
        return TaskMMOEDANN(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'fe_mmoe_ddc':
        return FEMMOEDDC(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'fe_mmoe_dann':
        return FEMMOEDANN(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'fe_mmoe_mcd':
        FE = FEMMOEMCD(num_classes, backbone_)
        C1 = ImageClassifier(backbone_[0].out_channels, num_classes)
        C2 = ImageClassifier(backbone_[0].out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'fe_mmoe_sol1':
        FE = FEMMOESol1(num_classes, backbone_)
        C1 = ImageClassifier(backbone_[0].out_channels, num_classes)
        C2 = ImageClassifier(backbone_[0].out_channels, num_classes)
        return FE, C1, C2
    # elif CFG.MODEL.NAME == 'dqn':
    #     return DQN(backbone_.out_channels, num_classes)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
