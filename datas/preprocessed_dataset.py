import os
import torch
import numpy as np

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
            self.data, self.gt = self.transform(self.data_path, self.gt)
        # print(len(self.data), self.data.shape, self.gt.shape)

    def __getitem__(self, item):
        return self.data[item], self.gt[item]

    def __len__(self):
        return len(self.data)

    @property
    def num_channels(self):
        return 48

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    @property
    def num_classes(self):
        return len(self.labels)

    @property
    def labels(self):
        # e.g. [0, 1, 2]
        return list(range(len(self.names)))

    @property
    def names(self):
        # e.g. ['background', 'road', 'building']
        return [
            'Healthy grass',
            'Stressed grass',
            'Synthetic grass',
            'Trees',
            'Soil',
            'Water',
            'Residential'
        ]
