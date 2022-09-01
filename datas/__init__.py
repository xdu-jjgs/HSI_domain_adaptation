import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawHouston, RawHyRANK
from datas.preprocessed_dataset import PreprocessedHouston


def build_transform():
    if CFG.DATASET.NAME in ['RAW_Houston', 'RAW_HyRANK']:
        # 对整个数据集处理
        # 归一化、裁剪
        transform = transforms.Compose([
            transforms.ZScoreNormalize(),
            transforms.CropImage((CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH), CFG.DATASET.PATCH.PAD_MODE,
                                 selector=lambda x, y: y != 0),
            transforms.LabelRenumber(-1),
            transforms.ToTensor()
        ])
    elif CFG.DATASET.NAME in ['PREPROCESSED_Houston', 'PREPROCESSED_HyRANK']:
        transform = transforms.Compose([
            transforms.DataAugment(ratio=CFG.DATASET.AUGMENT.RATIO, trans=CFG.DATASET.AUGMENT.TRANS)
        ])
    else:
        raise NotImplementedError('invalid dataset: {} for transform'.format(CFG.DATASET.NAME))
    return transform


def build_dataset(split: str):
    # assert split in ['train', 'val', 'test']
    if CFG.DATASET.NAME == 'RAW_Houston':
        dataset = RawHouston(CFG.DATASET.ROOT, split, transform=build_transform())
    elif CFG.DATASET.NAME == 'RAW_HyRANK':
        dataset = RawHyRANK(CFG.DATASET.ROOT, split, transform=build_transform())
    elif CFG.DATASET.NAME == 'PREPROCESSED_Houston':
        if split == 'train':
            dataset = PreprocessedHouston(CFG.DATASET.ROOT, split, transform=build_transform())
        else:
            dataset = PreprocessedHouston(CFG.DATASET.ROOT, split)
    elif CFG.DATASET.NAME == 'PREPROCESSED_HyRANK':
        if split == 'train':
            dataset = PreprocessedHouston(CFG.DATASET.ROOT, split, transform=build_transform())
        else:
            dataset = PreprocessedHouston(CFG.DATASET.ROOT, split)
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
