import torch.optim as optim

from configs import CFG


def build_optimizer(model):
    if CFG.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              CFG.OPTIMIZER.LR,
                              momentum=CFG.OPTIMIZER.MOMENTUM,
                              weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    elif CFG.OPTIMIZER.NAME == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               CFG.OPTIMIZER.LR,
                               weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError('invalid optimizer: {}'.format(CFG.OPTIMIZER.NAME))
    return optimizer
