from configs import CFG
from .focal import FocalLoss
from .compose import LossComposer
from .ce import CELoss, SoftmaxCELoss
from .bce import BCELoss, SigmoidBCELoss
from .mmd import MMDLoss, MultiKernelMMD
from .dice import DiceLoss, SigmoidDiceLoss


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
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion


def build_criterion():
    loss_names = CFG.CRITERION.ITEMS
    weights = CFG.CRITERION.WEIGHTS
    items = list(map(lambda x: build_loss(x), loss_names))
    composer = LossComposer(items, weights)
    return composer
