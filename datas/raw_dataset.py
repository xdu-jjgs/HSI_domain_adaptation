import os
import h5py
import numpy as np
import scipy.io as sio

from datas.base import Dataset


class RawHouston(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(RawHouston, self).__init__()
        assert split in ['train', 'val', 'test']
        if split == 'train':
            data_filename = 'Houston13.mat'
            gt_filename = 'Houston13_7gt.mat'
            gt_key = 'houston13new1_7'
        else:
            # 验证集等于测试集
            data_filename = 'Houston18.mat'
            gt_filename = 'Houston18_7gt.mat'
            gt_key = 'houston18new1'
        self.data_path = os.path.join(root, data_filename)
        self.data = sio.loadmat(self.data_path)['ori_data']
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.data_path)[gt_key]

        self.transform = transform
        if self.transform is not None:
            self.data, self.gt = self.transform(self.data_path, self.gt)

    def __getitem__(self, item):
        return self.data[item], self.gt[item]

    def __len__(self):
        return self.data['label'].shape[0]

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def num_channels(self):
        raise NotImplementedError('num_channels() not implemented')

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(8))

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
        return [
            'Unclassified',
            'Healthy grass',
            'Stressed grass',
            'Synthetic grass',
            'Trees',
            'Soil',
            'Water',
            'Residential'
        ]
