from typing import List
from configs import CFG
from .focal import FocalLoss
from .coral import CoralLoss
from .dis import Discrepancy
from .compose import LossComposer
from .ce import CELoss, SoftmaxCELoss
from .bce import BCELoss, SigmoidBCELoss
from .dice import DiceLoss, SigmoidDiceLoss
from .mmd import MMDLoss, LocalMMDLoss, JointMMDLoss


def build_loss(name):
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
    elif name == 'dis':
        criterion = Discrepancy()
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion


def build_criterion(loss_names: List[str] = None, weights: List[float] = None):
    if not loss_names:
        loss_names = CFG.CRITERION.ITEMS
    if not weights:
        weights = CFG.CRITERION.WEIGHTS
    items = list(map(lambda x: build_loss(x), loss_names))
    composer = LossComposer(items, weights)
    return composer
