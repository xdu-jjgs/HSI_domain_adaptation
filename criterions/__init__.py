from configs import CFG
from .mmd import MMDLoss, LocalMMDLoss, JointMMDLoss
from .focal import FocalLoss
from .coral import CoralLoss
from .compose import LossComposer
from .ce import CELoss, SoftmaxCELoss
from .bce import BCELoss, SigmoidBCELoss
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
    elif name == 'localmmd':
        criterion = LocalMMDLoss(kernel_num=CFG.CRITERION.KERNEL_NUM)
    elif name == 'jmmd':
        criterion = JointMMDLoss(kernel_num=CFG.CRITERION.KERNEL_NUM)
    elif name == 'coral':
        criterion = CoralLoss()
    else:
        raise NotImplementedError('invalid criterion: {}'.format(name))
    return criterion


def build_criterion(split: str = 'train'):
    # TODO: Find a better way
    if split == 'train':
        loss_names = CFG.CRITERION.ITEMS
        weights = CFG.CRITERION.WEIGHTS
    else:
        loss_names = ['softmax+ce']
        weights = [1.0]
    items = list(map(lambda x: build_loss(x), loss_names))
    composer = LossComposer(items, weights)
    return composer
