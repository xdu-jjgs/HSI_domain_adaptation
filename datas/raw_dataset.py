import os
import scipy.io as sio

from datas.base import Dataset


class RawHouston(Dataset):
    def __init__(self, root, split: str, transform=None):
        super(RawHouston, self).__init__()
        assert split in ['train', 'val', 'test']
        if split == 'train':
            data_filename = 'Houston13.mat'
            gt_filename = 'Houston13_7gt.mat'
        else:
            # 验证集等于测试集
            data_filename = 'Houston18.mat'
            gt_filename = 'Houston18_7gt.mat'
        self.data_path = os.path.join(root, data_filename)
        self.data = sio.loadmat(self.data_path)['ori_data'].astype('float16')
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.gt_path)['map'].astype('int')

        self.transform = transform
        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

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
        return list(range(len(self.names)))

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
