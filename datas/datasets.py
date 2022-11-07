import os
import scipy.io as sio

from typing import Tuple
from collections import Counter

from datas.base import Dataset


class HoustonDataset(Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(HoustonDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)
        data_filename = 'DataCube_ShanghaiHangzhou.mat'
        self.data_path = os.path.join(root, data_filename)
        if split == 'train':
            data_filename = 'Houston13.mat'
            gt_filename = 'Houston13_7gt.mat'
        else:
            # 验证集等于测试集
            data_filename = 'Houston18.mat'
            gt_filename = 'Houston18_7gt.mat'
        self.data_path = os.path.join(root, data_filename)
        # N*W*C
        self.data = sio.loadmat(self.data_path)['ori_data'].astype('float32')
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.gt_path)['map'].astype('int')

        self.selector = lambda x, y: y != 0
        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    @property
    def num_channels(self):
        return 48

    @property
    def names(self):
        return [
            'Healthy grass',
            'Stressed grass',
            'Trees',
            'Water',
            'Residential buildings',
            'Non-residential buildings',
            'Road'
        ]


class HyRankDataset(Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(HyRankDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)
        if split == 'train':
            data_filename = 'Dioni.mat'
            gt_filename = 'Dioni_gt.mat'
        else:
            # 验证集等于测试集
            data_filename = 'Loukia.mat'
            gt_filename = 'Loukia_gt.mat'
        self.data_path = os.path.join(root, data_filename)
        self.data = sio.loadmat(self.data_path)['ori_data'].astype('float32')
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.gt_path)['map'].astype('int')

        self.selector = lambda x, y: y not in [0, 6, 8]
        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    @property
    def num_channels(self):
        return 176

    @property
    def names(self):
        return [
            'Dense urban fabric',
            'Mineral extraction sites',
            'Non irrigated land',
            'Fruit trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense Vegetation',
            'Sparce Vegetation',
            'Sparce Areas',
            'Rocks and Sand',
            'Water',
            'Coastal Water'
        ]


class ShangHangDataset(Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(ShangHangDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)

        data_filename = 'DataCube_ShanghaiHangzhou.mat'
        self.data_path = os.path.join(root, data_filename)
        raw = sio.loadmat(self.data_path)
        if split == 'train':
            self.data = raw['DataCube2'].astype('float32')
            self.gt = raw['gt2'].astype('int')
        else:
            self.data = raw['DataCube1'].astype('float32')
            self.gt = raw['gt1'].astype('int')

        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    @property
    def num_channels(self):
        return 198

    @property
    def names(self):
        return [
            'Water',
            'Land/ Building',
            'Plant'
        ]
