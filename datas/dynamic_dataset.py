from datas.base import Dataset
import numpy as np
from typing import Tuple
import os
import scipy.io as sio
import torch


class HSIDataset(Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(HSIDataset, self).__init__()
        assert split in ['train', 'val', 'test']
        assert sample_num is None or sample_num >= 0
        assert sample_order is None or sample_order in ['', 'sequence', 'average']
        assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1, 'window size should be odd!'
        self.root = root
        self.split = split
        self.window_size = window_size
        self.pad_mode = pad_mode
        self.sample_num = sample_num
        self.sample_order = sample_order
        self.data = None
        self.gt = None
        self.gtmap = None
        self.selector = None
        self.coordinates = None
        self.pred = None
        self.transform = transform

    def cube_data(self):
        h, w, c = self.data.shape
        patch = ((self.window_size[0] - 1) // 2, (self.window_size[1] - 1) // 2)
        self.data = np.pad(self.data, (patch, patch, (0, 0)), self.pad_mode)
        coordinates = []
        gts = []
        for i in range(h):
            for j in range(w):
                if self.selector is None or self.selector(self.data[i, j], self.gt[i, j]):
                    coordinates.append([i, j])
                    gts.append(self.gt[i, j])
        coordinates = np.array(coordinates)
        gts = np.array(gts)
        return coordinates, gts

    def sample_data(self):
        label_unique = list(np.unique(self.gt))
        num_pre_class = self.sample_num // len(label_unique)
        count = [0] * len(label_unique)
        count_all = 0
        if self.sample_order == 'average':
            shuffle_index = np.random.permutation(np.arange(len(self.gt)))
            self.coordinates = self.coordinates[shuffle_index]
            self.gt = self.gt[shuffle_index]

        coordinates = []
        gts = []
        for ind, ele in enumerate(self.gt):
            if (self.sample_order == 'average' and count[ele] < num_pre_class) \
                    or (self.sample_order == 'sequence' and count_all < self.sample_num):
                gts.append(ele)
                coordinates.append(self.coordinates[ind])
                count[ele] += 1
                count_all += 1
            elif count_all >= self.sample_num:
                break
        else:
            if count_all < self.sample_num:
                if self.sample_order == 'average':
                    for ind, ele in enumerate(count):
                        if ele < num_pre_class:
                            print(
                                "Insufficient sample number for class {} in {} dataset,"
                                " expect {} but actually {}.".format(self.names[ind], self.split, num_pre_class, ele))
                raise IndexError(
                    "Insufficient sample number, expect total {}, actually {}".format(self.sample_num, count_all))
        return coordinates, gts

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, item):
        x1 = self.coordinates[item][0]
        y1 = self.coordinates[item][1]
        data = self.data[..., x1:x1 + self.window_size[0], y1:y1 + self.window_size[1]]
        gt = self.gt[item]
        return data, gt, item

    def update_pred(self, index, pred):
        x1 = self.coordinates[index, 0]
        y1 = self.coordinates[index, 1]
        self.pred[x1, y1] = pred+1

    def update_gtmap(self, index, gt):
        x1 = self.coordinates[index, 0]
        y1 = self.coordinates[index, 1]
        self.gtmap[x1, y1] = gt+1

    def get_pred(self):
        return self.pred

    def get_gtmap(self):
        return self.gtmap

    def name2label(self, name):
        return self.names.index(name)

    def label2name(self, label):
        return self.names[label]

    def get_labels(self):
        return self.gt

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
        raise NotImplementedError('names() not implemented')


class HoustonDataset(HSIDataset):
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
        h, w, c = self.data.shape

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

        self.pred = np.zeros([h, w])
        self.gtmap = np.zeros([h, w])

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


class HyRankDataset(HSIDataset):
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
        h, w, c = self.data.shape

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

        self.pred = np.zeros([h, w])
        self.gtmap = np.zeros([h, w])

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


class ShangHangDataset(HSIDataset):
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
        h, w, c = self.data.shape

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

        self.pred = np.zeros([h, w])
        self.gtmap = np.zeros([h, w])

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


class DynamicDataset(Dataset):
    def __init__(self):
        super(DynamicDataset, self).__init__()
        self.data = []
        self.gt = []
        self.confid = []
        self.data = torch.tensor(self.data)
        self.gt = torch.tensor(self.gt)
        self.confid = torch.tensor(self.confid)

    def __getitem__(self, index):
        return self.data[index, :, :, :], self.gt[index]

    def __len__(self):
        return len(self.data)

    def reshape(self):
        self.gt = self.gt.numpy().reshape(-1, )

    def get_labels(self):
        return self.gt

    def get_confid(self):
        return self.confid

    def append(self, data, gt, confid):
        self.data = torch.cat((self.data, data), dim=0)
        self.gt = torch.cat((self.gt, gt), dim=0)
        self.confid = torch.cat((self.confid, confid), dim=0)

    def flush(self):
        self.data = []
        self.gt = []
        self.confid = []
        self.data = torch.tensor(self.data)
        self.gt = torch.tensor(self.gt)
        self.confid = torch.tensor(self.confid)

    def print_info(self):
        print(self.data.size())
        print(self.gt.size())
        print(self.confid.size())
