import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawHouston
from datas.preprocessed_dataset import PreprocessedHouston


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
    elif CFG.DATASET.NAME == 'PRRPROCESSED_Houston':
        dataset = PreprocessedHouston(CFG.DATASET.ROOT, split)
    else:
        raise NotImplementedError('invalid dataset: {} for dataset'.format(CFG.DATASET.NAME))
    return dataset


def build_dataloader(dataset, split: str, sampler=None):
    assert split in ['train', 'val', 'test']
    if split == 'train':
        return DataLoader(dataset,
                          batch_size=CFG.DATALOADER.BATCH_SIZE // dist.get_world_size(),
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler
                          )
    elif split == 'val':
        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler,
                          )
    elif split == 'test':
        return DataLoader(dataset,
                          batch_size=1,
                          num_workers=CFG.DATALOADER.NUM_WORKERS,
                          pin_memory=True if CFG.DATALOADER.NUM_WORKERS > 0 else False,
                          sampler=sampler
                          )
