import torch.optim as optim

from configs import CFG


def build_scheduler(optimizer):
    if CFG.SCHEDULER.NAME == '':
        # scheduler is allowed to be None, which means the learning rate wouldn't be changed during training
        scheduler = None
    elif CFG.SCHEDULER.NAME == 'poly':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=CFG.SCHEDULER.GAMMA,
                                                     verbose=True)
    elif CFG.SCHEDULER.NAME == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=CFG.SCHEDULER.MODE,
                                                         factor=CFG.SCHEDULER.FACTOR,
                                                         patience=CFG.SCHEDULER.PATIENCE,
                                                         verbose=True)
    else:
        raise NotImplementedError('invalid scheduler: {}'.format(CFG.SCHEDULER.NAME))
    return scheduler
