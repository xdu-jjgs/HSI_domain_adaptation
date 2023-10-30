import os
import logging
import numpy as np
import datas.transforms as transforms

from PIL import Image

from configs import CFG
from datas.datasets import HoustonDataset, HyRankDataset, ShangHangDataset

transform = transforms.Compose([
    transforms.LabelRenumber(),
    transforms.ToTensor(),
])


def stretch_rgb2png(rgb, lower_percent=0.5, higher_percent=99.5):
    low_values = np.percentile(rgb, lower_percent, axis=(0, 1))
    high_values = np.percentile(rgb, higher_percent, axis=(0, 1))
    stretched_rgb = (rgb - low_values) / (high_values - low_values) * 255
    png = np.clip(stretched_rgb, 0, 255).astype(np.uint8)
    return png


def build_dataset(split: str):
    assert split in ['train', 'test']
    if CFG.DATASET.NAME == 'Houston':
        dataset = HoustonDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                 CFG.DATASET.PATCH.PAD_MODE, transform=transform)
    elif CFG.DATASET.NAME == 'HyRANK':
        dataset = HyRankDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                CFG.DATASET.PATCH.PAD_MODE, transform=transform)
    elif CFG.DATASET.NAME == 'ShangHang':
        dataset = ShangHangDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                   CFG.DATASET.PATCH.PAD_MODE, transform=transform)
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


CFG.merge_from_file(r'./hyrank/save_rgb_image.yaml')
rgb_bands = [22, 10, 6]
source_dataset = build_dataset('train')
target_dataset = build_dataset('test')
val_dataset = target_dataset
assert source_dataset.num_classes == val_dataset.num_classes
logging.info("Number of train {}, val {}, test {}".format(len(source_dataset), len(val_dataset), len(target_dataset)))
NUM_CHANNELS = source_dataset.num_channels
NUM_CLASSES = source_dataset.num_classes
logging.info("Number of class: {}".format(NUM_CLASSES))

lower_percent, higher_percent = 0.5, 99.5

img = np.array(source_dataset.data[rgb_bands, ...].permute(1, 2, 0))
low_values = np.percentile(img, lower_percent, axis=(0, 1))
high_values = np.percentile(img, higher_percent, axis=(0, 1))
stretched_rgb = (img - low_values) / (high_values - low_values) * 255
png = np.clip(stretched_rgb, 0, 255).astype(np.uint8)
Image.fromarray(png).save(r'E:\zts\dataset\hyrank_rgb\source_dataset.png')
for ind in range(len(source_dataset)):
    x_s, label_s = source_dataset[ind]
    img = np.array(x_s[rgb_bands, ...].permute(1, 2, 0))
    name = source_dataset.label2name(label_s)
    stretched_rgb = (img - low_values) / (high_values - low_values) * 255
    png = np.clip(stretched_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(png).save(r'E:\zts\dataset\hyrank_rgb\train\{}_{}.png'.format(ind + 1, name))

img = np.array(target_dataset.data[rgb_bands, ...].permute(1, 2, 0))
low_values = np.percentile(img, lower_percent, axis=(0, 1))
high_values = np.percentile(img, higher_percent, axis=(0, 1))
stretched_rgb = (img - low_values) / (high_values - low_values) * 255
png = np.clip(stretched_rgb, 0, 255).astype(np.uint8)
Image.fromarray(png).save(r'E:\zts\dataset\hyrank_rgb\target_dataset.png')
for ind in range(len(target_dataset)):
    x_t, label_t = source_dataset[ind]
    img = np.array(x_t[rgb_bands, ...].permute(1, 2, 0))
    name = source_dataset.label2name(label_t)
    stretched_rgb = (img - low_values) / (high_values - low_values) * 255
    png = np.clip(stretched_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(png).save(r'E:\zts\dataset\hyrank_rgb\test\{}_{}.png'.format(ind + 1, name))
