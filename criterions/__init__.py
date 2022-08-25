from configs import CFG
from .bce import BCELoss, SigmoidBCELoss
from .ce import CELoss, SoftmaxCELoss
from .dice import DiceLoss, SigmoidDiceLoss
from .focal import FocalLoss


def build_criterion():
    if CFG.CRITERION.NAME == 'ce':
        criterion = CELoss()
    elif CFG.CRITERION.NAME == 'softmax+ce':
        criterion = SoftmaxCELoss()
    elif CFG.CRITERION.NAME == 'bce':
        criterion = BCELoss()
    elif CFG.CRITERION.NAME == 'sigmoid+bce':
        criterion = SigmoidBCELoss()
    elif CFG.CRITERION.NAME == 'dice':
        criterion = DiceLoss()
    elif CFG.CRITERION.NAME == 'sigmoid+dice':
        criterion = SigmoidDiceLoss()
    elif CFG.CRITERION.NAME == 'focal':
        criterion = FocalLoss()
    else:
        raise NotImplementedError('invalid criterion: {}'.format(CFG.CRITERION.NAME))
    return criterion
