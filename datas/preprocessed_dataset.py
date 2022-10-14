import os
import torch
import numpy as np
import scipy.io as sio

from typing import Tuple

from datas.base import Dataset


class PreprocessedHouston(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(PreprocessedHouston, self).__init__()
        assert split in ['train', 'val', 'test']

        self.data_path = os.path.join(root, '{}_data.pt'.format(split))
        self.gt_path = os.path.join(root, '{}_gt.npy'.format(split))

        # NCHW
        self.data = torch.load(self.data_path)
        self.gt = np.load(self.gt_path)

        self.transform = transform
        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)
        # print(len(self.data), self.data.shape, self.gt.shape)

    def __getitem__(self, item):
        return self.data[item], self.gt[item]

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return self.gt

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    @property
    def num_channels(self):
        return 48

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
        return [
            'Healthy grass',
            'Stressed grass',
            'Trees',
            'Water',
            'Residential buildings',
            'Non-residential buildings',
            'Road'
        ]


class PreprocessedHyRank(PreprocessedHouston):
    @property
    def num_channels(self):
        return 176

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
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


class PreprocessedShangHang(Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, ratio: int = 1, transform=None):
        super(PreprocessedShangHang, self).__init__()
        assert split in ['train', 'val', 'test']
        assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1, 'window size should be odd!'
        self.split = split
        self.window_size = window_size
        self.pad_mode = pad_mode
        self.ratio = ratio

        data_filename = 'DataCube_ShanghaiHangzhou.mat'
        self.data_path = os.path.join(root, data_filename)
        raw = sio.loadmat(self.data_path)
        if split == 'train':
            self.data = raw['DataCube2'].astype('float32')
        else:
            self.data = raw['DataCube1'].astype('float32')
        self.gt_path = os.path.join(root, '{}_gt.npy'.format(split))
        self.gt = np.load(self.gt_path)
        self.coordinate_path = os.path.join(root, '{}_coordinate.npy'.format(split))
        self.coordinate = np.load(self.coordinate_path)

        patch = ((self.window_size[0] - 1) // 2, (self.window_size[1] - 1) // 2)
        self.data = np.pad(self.data, (patch, patch, (0, 0)), self.pad_mode)

        self.transform = transform
        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

    def __getitem__(self, item):
        n_ori = self.coordinate.shape[0]
        x1 = self.coordinate[item % n_ori][0]
        y1 = self.coordinate[item % n_ori][1]
        data = self.data[..., x1:x1 + self.window_size[0], y1:y1 + self.window_size[1]]
        gt = self.gt[item % n_ori]
        return data, gt

    def __len__(self):
        if self.split == 'train':
            return self.gt.shape[0] * self.ratio
        else:
            return self.gt.shape[0]

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    @property
    def num_channels(self):
        return 198

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
        return [
            'Water',
            'Land/ Building',
            'Plant'
        ]
