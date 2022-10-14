from configs import CFG
# import torch.nn as nn
from .mmd import MMDLoss, LocalMMDLoss
from .focal import FocalLoss
from .coral import CoralLoss
from .compose import LossComposer
from .ce import CELoss, SoftmaxCELoss
from .bce import BCELoss, SigmoidBCELoss
from .dice import DiceLoss, SigmoidDiceLoss
from .dis import Discrepancy
from .class_weight_ce import ClassWeightCE
from .confid_weight_ce import ConfidWeightCE
from .capsuleloss import MarginLoss


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
    elif name == 'coral':
        criterion = CoralLoss()
    elif name == 'dis':
        criterion = Discrepancy()
    elif name == 'class_weight_ce':
        criterion = ClassWeightCE()
    elif name == 'confid_weight_ce':
        criterion = ConfidWeightCE()
    elif name == 'margin':
        criterion = MarginLoss()
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion


def build_criterion(loss_names=None, weights=None):
    # TODO: change the split into loss_name directly
    # loss_name = list['loss_name','loss_name'] defined in CFG.CRITERION.ITEMS
    items = list(map(lambda x: build_loss(x), loss_names))
    composer = LossComposer(items, weights)
    return composer
