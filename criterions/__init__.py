from configs import CFG
from .focal import FocalLoss
from .coral import CoralLoss
from .dis import L1Distance
from .ce import CELoss, SoftmaxCELoss
from .bce import BCELoss, SigmoidBCELoss
from .dice import DiceLoss, SigmoidDiceLoss
from .mmd import MMDLoss, LocalMMDLoss, JointMMDLoss
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
    elif name == 'softmax+ce+ls':
        criterion = ConfidenceBasedSelfTrainingLoss(threshold=CFG.CRITERION.THRESHOLD)
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion
