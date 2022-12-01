import torch.nn as nn

from configs import CFG
from .focal import FocalLoss
from .coral import CoralLoss
from .ce import CELoss, SoftmaxCELoss, Entropy
from .bce import BCELoss, SigmoidBCELoss
from .dice import DiceLoss, SigmoidDiceLoss
from .mmd import MMDLoss, LocalMMDLoss, JointMMDLoss
from .dis import L1Distance, L2Distance, SoftmaxL1Distance
from tllib.self_training.dst import WorstCaseEstimationLoss
from tllib.self_training.pseudo_label import ConfidenceBasedSelfTrainingLoss



def build_criterion(name):
    if name == 'ce':
        criterion = CELoss()
    elif name == 'softmax+ce':
        criterion = SoftmaxCELoss()
    elif name == 'bce':
        criterion = BCELoss()
    elif name == 'sigmoid+bce':
        criterion = SigmoidBCELoss()
    elif name == 'dice':
        criterion = DiceLoss()
    elif name == 'sigmoid+dice':
        criterion = SigmoidDiceLoss()
    elif name == 'focal':
        criterion = FocalLoss()
    elif name == 'mmd':
        criterion = MMDLoss(kernel_num=CFG.CRITERION.KERNEL_NUM)
    elif name == 'localmmd':
        criterion = LocalMMDLoss(kernel_num=CFG.CRITERION.KERNEL_NUM)
    elif name == 'jmmd':
        criterion = JointMMDLoss(kernel_num=CFG.CRITERION.KERNEL_NUM)
    elif name == 'coral':
        criterion = CoralLoss()
    elif name == 'l1dis':
        criterion = L1Distance()
    elif name == 'l2dis':
        criterion = L2Distance()
    elif name == 'softmax+l1dis':
        criterion = SoftmaxL1Distance()
    elif name == 'entropy':
        criterion = Entropy()
    elif name == 'kldiv':
        criterion = nn.KLDivLoss()
    elif name == 'softmax+ce+ls':
        criterion = ConfidenceBasedSelfTrainingLoss(threshold=CFG.CRITERION.THRESHOLD)
    elif name == 'wcec':
        return WorstCaseEstimationLoss(CFG.HYPERPARAMS[0])
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion
