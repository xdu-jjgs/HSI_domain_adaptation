import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawHouston


def build_transform():
    if CFG.DATASET.NAME == 'RAW_Houston':
        # 对整个数据集处理
        # 归一化、裁剪
        transform = transforms.Compose([
            transforms.ZScoreNormalize(),
            transforms.CropImage((CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH), CFG.DATASET.PATCH.PAD_MODE),
            transforms.ToTensor()
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    # assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'RAW_Houston':
        dataset = RawHouston(CFG.DATASET.ROOT, split, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset
