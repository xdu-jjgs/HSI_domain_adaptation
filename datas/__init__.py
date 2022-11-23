import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader
from tllib.utils.data import ForeverDataIterator

from configs import CFG
from datas.base import DynamicDataset
from datas.datasets import HoustonDataset, HyRankDataset, ShangHangDataset


def build_transform():
    if CFG.DATASET.NAME in ['Houston', 'HyRANK', 'ShangHang']:
        transform = transforms.Compose([
            transforms.LabelRenumber(),
            transforms.ZScoreNormalize(),
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    assert split in ['train', 'val', 'test', 'dynamic']
    if split == 'dynamic':
        return DynamicDataset()
    if CFG.DATASET.NAME == 'Houston':
        dataset = HoustonDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                 CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                 CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'HyRANK':
        dataset = HyRankDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    elif CFG.DATASET.NAME == 'ShangHang':
        dataset = ShangHangDataset(CFG.DATASET.ROOT, split, (CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH),
                                   CFG.DATASET.PATCH.PAD_MODE, CFG.DATASET.SAMPLE_NUM if split == 'train' else None,
                                   CFG.DATASET.SAMPLE_ORDER if split == 'train' else None, transform=build_transform())
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


def build_dataloader(dataset, sampler=None):
    return DataLoader(dataset,
                      batch_size=CFG.DATALOADER.BATCH_SIZE // dist.get_world_size(),
                      num_workers=CFG.DATALOADER.NUM_WORKERS,
                      pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                      sampler=sampler,
                      drop_last=True
                      )


def build_iterator(dataloader: DataLoader):
    return ForeverDataIterator(dataloader)
