import numpy as np

from typing import Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


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
        self.selector = None
        self.coordinates = None
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
        return data, gt

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
        raise NotImplementedError('names() not implemented')

    @property
    def pixels(self):
        raise NotImplementedError('pixels() not implemented')


class DynamicDataset(Dataset):
    def __init__(self):
        self.data = []
        self.gt = []

    def __getitem__(self, index) -> T_co:
        return self.data[index], self.gt[index]

    def __len__(self):
        return len(self.data)

    def append(self, data, gt):
        self.data.append(data)
        self.gt.append(gt)

    def flush(self):
        self.data = []
        self.gt = []