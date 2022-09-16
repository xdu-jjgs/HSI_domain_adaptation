import torch.optim as optim

from configs import CFG


def build_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr,
                              momentum=CFG.OPTIMIZER.MOMENTUM,
                              weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr,
                               weight_decay=CFG.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError('invalid optimizer: {}'.format(optimizer_name))
    return optimizer
