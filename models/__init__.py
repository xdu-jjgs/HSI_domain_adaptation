from .ddc import DDC
from .dqn import DQN
from .dst import DST
from .dsn import (DSN, DSN_Gate, DSN_INN, DSN_INN_Gate, DSN_NoDecoder, DSN_INN_NoDecoder,
                  DSN_NoDecoder_Nospec, DSN_INN_NoDecoder_Nospec)
from .hma import INN, INNDANN
from .dann import DANN
from .vdd import VDD, VDDFixed
from .dadst import DADST, DADASTMapping, DADSTFFT
from .task_mmoe import TaskMMOEDDC, TaskMMOEDANN
from .fe_param import FEPARAMDDC, FEPARAMDANN, FEPARAMMCD
from .fe_mmoe import FEMMOEDDC, FEMMOEDANN, FEMMOEMCD, FEMMOESol1
from .dd_fe_mmoe import DDFEMMOE
from .de_fe_mmoe import DEFEMMOEDANN, DEFEMMOEDADST, DEFEMMOEDADST_GateConv, DEFEMMOEDADST_Mapping, DEFEMMOEDADST_Shared

from configs import CFG
from models.backbone import build_backbone, ImageClassifier


def build_model(num_channels, num_classes):
    # build backbone/experts
    if CFG.MODEL.BACKBONE:
        backbone_ = build_backbone(num_channels, CFG.MODEL.BACKBONE)
    elif CFG.MODEL.EXPERTS[0]:
        backbone_ = [build_backbone(num_channels, i) for i in CFG.MODEL.EXPERTS]
    else:
        raise NotImplementedError('invalid backbone: {} or experts: {}'.format(CFG.MODEL.BACKBONE, CFG.MODEL.EXPERTS))
    # build model
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
    elif CFG.MODEL.NAME == 'dadst':
        return DADST(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dadst_mapping':
        return DADASTMapping(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dadst_fft':
        return DADSTFFT(num_classes, backbone_)
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
    elif CFG.MODEL.NAME == 'fe_param_ddc':
        return FEPARAMDDC(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'fe_param_dann':
        return FEPARAMDANN(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'fe_param_mcd':
        FE = FEPARAMMCD(num_classes, backbone_)
        C1 = ImageClassifier(backbone_[0].out_channels, num_classes)
        C2 = ImageClassifier(backbone_[0].out_channels, num_classes)
        return FE, C1, C2
    elif CFG.MODEL.NAME == 'dd_fe_mmoe':
        return DDFEMMOE(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'de_fe_mmoe_dann':
        return DEFEMMOEDANN(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'de_fe_mmoe_dadst':
        return DEFEMMOEDADST(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'de_fe_mmoe_dadst_gate_conv':
        return DEFEMMOEDADST_GateConv(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'de_fe_mmoe_dadst_mapping':
        return DEFEMMOEDADST_Mapping(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'de_fe_mmoe_dadst_shared':
        return DEFEMMOEDADST_Shared(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'vdd':
        return VDD(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'vdd_fixed':
        return VDDFixed(num_classes, backbone_)
    elif CFG.MODEL.NAME == 'dsn':
        return DSN(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_gate':
        return DSN_Gate(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_inn':
        return DSN_INN(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_inn_gate':
        return DSN_INN_Gate(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_nodecoder':
        return DSN_NoDecoder(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_nodecoder_nospec':
        return DSN_NoDecoder_Nospec(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_inn_nodecoder':
        return DSN_INN_NoDecoder(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'dsn_inn_nodecoder_nospec':
        return DSN_INN_NoDecoder_Nospec(num_classes, backbone_, CFG.DATASET.PATCH.WIDTH)
    elif CFG.MODEL.NAME == 'hma_ddc':
        FE = backbone_
        inn = INN(in_channels=backbone_.out_channels, num_block=CFG.HYPERPARAMS[0])
        C = ImageClassifier(backbone_.out_channels, num_classes)
        return FE, inn, C
    elif CFG.MODEL.NAME == 'hma_dann':
        FE = backbone_
        inn = INNDANN(in_channels=backbone_.out_channels, num_block=CFG.HYPERPARAMS[0])
        C = ImageClassifier(backbone_.out_channels, num_classes)
        return FE, inn, C
    # elif CFG.MODEL.NAME == 'dqn':
    #     return DQN(backbone_.out_channels, num_classes)
    raise NotImplementedError('invalid model: {}'.format(CFG.MODEL.NAME))
