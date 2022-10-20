from yacs.config import CfgNode as CN

_C = CN()

_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.ROOT = ''
_C.DATASET.PATCH = CN()
_C.DATASET.PATCH.HEIGHT = 0
_C.DATASET.PATCH.WIDTH = 0
_C.DATASET.PATCH.PAD_MODE = ''
_C.DATASET.AUGMENT = CN()
_C.DATASET.AUGMENT.RATIO = 1
_C.DATASET.AUGMENT.TRANS = ''

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 0
_C.DATALOADER.NUM_WORKERS = 0

_C.MODEL = CN()
_C.MODEL.NAME = ''

_C.CRITERION = CN()
_C.CRITERION.ITEMS = [''] * 10
_C.CRITERION.WEIGHTS = [0.] * 10
_C.CRITERION.KERNEL_NUM = 0
_C.CRITERION.THRESHOLD = 0.

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = ''
_C.OPTIMIZER.LR = 0.
_C.OPTIMIZER.MOMENTUM = 0.
_C.OPTIMIZER.WEIGHT_DECAY = 0.

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = ''
_C.SCHEDULER.GAMMA = 0.
_C.SCHEDULER.MODE = ''
_C.SCHEDULER.FACTOR = 0.
_C.SCHEDULER.PATIENCE = 0

_C.EPOCHS = 0
_C.EPOCHFE = 0

CFG = _C.clone()
CFG.freeze()
