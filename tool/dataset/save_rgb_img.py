import cv2
import logging

from configs import CFG
from datas.datasets import HoustonDataset, HyRankDataset, ShangHangDataset


def build_dataset(split: str):
    assert split in ['train', 'test']
    if CFG.DATASET.NAME == 'Houston':
        dataset = HoustonDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                 CFG.DATASET.PATCH.PAD_MODE, transform=None)
    elif CFG.DATASET.NAME == 'HyRANK':
        dataset = HyRankDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                CFG.DATASET.PATCH.PAD_MODE, transform=None)
    elif CFG.DATASET.NAME == 'ShangHang':
        dataset = ShangHangDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                   CFG.DATASET.PATCH.PAD_MODE, transform=None)
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


source_dataset = build_dataset('train')
target_dataset = build_dataset('test')
val_dataset = target_dataset
assert source_dataset.num_classes == val_dataset.num_classes
logging.info("Number of train {}, val {}, test {}".format(len(source_dataset), len(val_dataset), len(target_dataset)))
NUM_CHANNELS = source_dataset.num_channels
NUM_CLASSES = source_dataset.num_classes
logging.info("Number of class: {}".format(NUM_CLASSES))

for ind in range(len(source_dataset)):
    x_s, label_s = source_dataset[ind]
    img = x_s[[23, 11, 7], :, :]
    name = source_dataset.labels[label_s]
    cv2.imwrite(r'\train\{}_{}.png'.format(ind + 1, name), img)

for ind in range(len(target_dataset)):
    x_s, label_s = source_dataset[ind]
    img = x_s[[23, 11, 7], :, :]
    name = source_dataset.labels[label_s]
    cv2.imwrite(r'\test\{}_{}.png'.format(ind + 1, name), img)