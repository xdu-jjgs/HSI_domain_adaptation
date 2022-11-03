import numpy as np
import torch.utils as utils

from typing import Tuple


class Dataset(utils.data.Dataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, transform=None):
        super(Dataset, self).__init__()
        assert split in ['train', 'val', 'test']
        assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1, 'window size should be odd!'
        self.root = root
        self.split = split
        self.window_size = window_size
        self.pad_mode = pad_mode
        self.data = None
        self.gt = None
        self.selector = None
        self.coordinates = None
        self.transform = transform

    def cube_image(self):
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
