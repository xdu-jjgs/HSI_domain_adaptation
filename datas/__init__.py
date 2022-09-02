import torch.distributed as dist
import datas.transforms as transforms

from torch.utils.data import DataLoader

from configs import CFG
from datas.raw_dataset import RawHouston, RawHyRANK, RawShangHang
from datas.preprocessed_dataset import PreprocessedHouston, PreprocessedHyRank, PreprocessedShangHang


def build_transform():
    if CFG.DATASET.NAME == 'RAW_Houston':
        # 对整个数据集处理
        transform = transforms.Compose([
            transforms.ZScoreNormalize(),
            transforms.CropImage((CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH), CFG.DATASET.PATCH.PAD_MODE,
                                 selector=lambda x, y: y != 0),
            transforms.LabelRenumber(),
            transforms.ToTensor()
        ])
    elif CFG.DATASET.NAME == 'RAW_HyRANK':
        # 对整个数据集处理
        transform = transforms.Compose([
            transforms.ZScoreNormalize(),
            transforms.CropImage((CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH), CFG.DATASET.PATCH.PAD_MODE,
                                 selector=lambda x, y: y not in [0, 6, 8]),
            transforms.LabelRenumber(),
            transforms.ToTensor()
        ])
    elif CFG.DATASET.NAME == 'RAW_ShangHang':
        # 对整个数据集处理
        transform = transforms.Compose([
            transforms.CropImage((CFG.DATASET.PATCH.HEIGHT, CFG.DATASET.PATCH.WIDTH), CFG.DATASET.PATCH.PAD_MODE,
                                 return_type='coordinate'),
            transforms.LabelRenumber(),
            transforms.ToTensor()
        ])

    elif CFG.DATASET.NAME in ['PREPROCESSED_Houston', 'PREPROCESSED_HyRANK']:
        transform = transforms.Compose([
            transforms.DataAugment(ratio=CFG.DATASET.AUGMENT.RATIO, trans=CFG.DATASET.AUGMENT.TRANS)
        ])
    elif CFG.DATASET.NAME == 'PREPROCESSED_ShangHang':
        transform = transforms.Compose([
            transforms.ZScoreNormalize(),
            transforms.ToTensor(),
            transforms.DataAugment(ratio=CFG.DATASET.AUGMENT.RATIO, trans=CFG.DATASET.AUGMENT.TRANS),

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
    elif CFG.DATASET.NAME == 'RAW_ShangHang':
        dataset = RawShangHang(CFG.DATASET.ROOT, split, transform=build_transform())

    elif CFG.DATASET.NAME == 'PREPROCESSED_Houston':
        dataset = PreprocessedHouston(CFG.DATASET.ROOT, split,
                                      transform=build_transform() if split == 'train' else None)
    elif CFG.DATASET.NAME == 'PREPROCESSED_HyRANK':
        dataset = PreprocessedHyRank(CFG.DATASET.ROOT, split,
                                     transform=build_transform() if split == 'train' else None)
    elif CFG.DATASET.NAME == 'PREPROCESSED_ShangHang':
        dataset = PreprocessedShangHang(CFG.DATASET.ROOT, split,
                                        transform=build_transform() if split == 'train' else None)
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
