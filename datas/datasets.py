import os
import scipy.io as sio

from typing import Tuple

from datas.base import HSIDataset


class HoustonDataset(HSIDataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(HoustonDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)
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
        self.gt_raw = self.gt.copy()

        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    def selector(self, x, y):
        return y != 0

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

    @property
    def pixels(self):
        return [
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105]
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
        self.gt_raw = self.gt.copy()

        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    def selector(self, x, y):
        return y not in [0, 6, 8]

    @property
    def num_channels(self):
        return 176

    @property
    def names(self):
        return [
            'Dense urban fabric',
            'Mineral extraction sites',
            'Non-irrigated arable land',
            'Fruit trees',
            'Olive Groves',
            'Coniferous Forest',
            'Dense scleroph, Vegetation',
            'Sparse scleroph, Vegetation',
            'Sparse vegetated areas',
            'Rocks & sand',
            'Water',
            'Coastal Water'
        ]

    @property
    def pixels(self):
        return [
            [141, 211, 199],
            [255, 255, 179],
            [190, 186, 218],
            [251, 128, 114],
            [128, 177, 211],
            [253, 180, 98],
            [179, 222, 105],
            [252, 205, 229],
            [217, 217, 217],
            [188, 128, 189],
            [204, 128, 189],
            [255, 237, 111]
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
        self.gt_raw = self.gt.copy()
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

    @property
    def pixels(self):
        return [
            [0, 157, 130],
            [255, 255, 85],
            [110, 101, 172]
        ]


class PaviaDataset(HSIDataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(PaviaDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)
        if split == 'train':
            data_filename = 'paviaU.mat'
            gt_filename = 'paviaU_7gt.mat'
        else:
            # 验证集等于测试集
            data_filename = 'paviaC.mat'
            gt_filename = 'paviaC_7gt.mat'
        self.data_path = os.path.join(root, data_filename)
        # N*W*C
        self.data = sio.loadmat(self.data_path)['ori_data'].astype('float32')
        self.gt_path = os.path.join(root, gt_filename)
        self.gt = sio.loadmat(self.gt_path)['map'].astype('int')
        self.gt_raw = self.gt.copy()

        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    def selector(self, x, y):
        return y != 0

    @property
    def num_channels(self):
        return 102

    @property
    def names(self):
        return [
            'Tree',
            'Asphalt',
            'Brick',
            'Bitumen',
            'Shadow',
            'Meadow',
            'Bare soil'
        ]

    @property
    def pixels(self):
        return [
            [0, 31, 255],
            [0, 175, 255],
            [63, 255, 191],
            [219, 255, 41],
            [255, 159, 0],
            [255, 15, 0],
            [128, 0, 0]
        ]


class IndianaDataset(HSIDataset):
    def __init__(self, root, split: str, window_size: Tuple[int, int], pad_mode: str, sample_num: int = None,
                 sample_order: str = None, transform=None):
        super(IndianaDataset, self).__init__(root, split, window_size, pad_mode, sample_num, sample_order, transform)

        data_filename = 'DataCube.mat'
        self.data_path = os.path.join(root, data_filename)
        raw = sio.loadmat(self.data_path)
        if split == 'train':
            self.data = raw['DataCube1'].astype('float32')
            self.gt = raw['gt1'].astype('int')
        else:
            self.data = raw['DataCube2'].astype('float32')
            self.gt = raw['gt2'].astype('int')
        self.gt_raw = self.gt.copy()
        self.coordinates, self.gt = self.cube_data()

        if self.transform is not None:
            self.data, self.gt = self.transform(self.data, self.gt)

        if self.sample_order:
            self.coordinates, self.gt = self.sample_data()

    def selector(self, x, y):
        return y != 0

    @property
    def num_channels(self):
        return 220

    @property
    def names(self):
        return [
            'Concrete / Asphalt',
            'Corn cleanTill',
            'Corn cleanTill EW',
            'Orchard',
            'Soybeans cleanTill',
            'Soybeans cleanTill EW',
            'Wheat'
        ]

    @property
    def pixels(self):
        return [
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 0, 255],
            [255, 255, 0],
            [128, 128, 255]
        ]
