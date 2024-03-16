import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from typing import List


class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, data, label):
        for transform in self.transforms:
            if data is not None:
                data, label = transform(data, label)
        return data, label


class RandomCrop(nn.Module):
    # 元素
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, data, label):
        if data is not None:
            h, w, _ = data.shape
            new_h, new_w = self.size

            top = np.random.randint(0, h - new_h)
            down = top + new_h
            left = np.random.randint(0, w - new_w)
            right = left + new_w

            if data is not None:
                data = data[top:down, left:right, :]
            if label is not None:
                label = label[top:down, left:right]

        return data, label


class ToTensor(nn.Module):
    # 元素/整体
    def forward(self, data, label):
        shape = data.shape
        data = torch.tensor(data, dtype=torch.float32)
        if len(shape) == 4:
            data = data.permute((0, 3, 1, 2))
        elif len(shape) == 3:
            data = data.permute((2, 0, 1))
        return data, label


class Normalize(nn.Module):
    # 整体/元素
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, data, label):
        data = self.normalize(data)
        return data, label


class LabelRenumber(nn.Module):
    # 整体/元素
    def forward(self, data, label):
        def renumber(ele):
            return label_cur.index(ele)

        renumber_np = np.frompyfunc(renumber, 1, 1)
        label_cur = list(np.unique(label))
        label = renumber_np(label).astype('int')

        return data, label


class ZScoreNormalize(nn.Module):
    # 整体
    def forward(self, data, label):
        h, w, c = data.shape
        data_type = data.dtype
        data = data.reshape(h * w, c).astype('float32')
        mean = np.mean(data, axis=0)
        # std上溢了，需要用float32
        std = np.std(data, axis=0)
        # print("mean:{}, std: {}".format(mean, std))
        data = (data - mean) / std
        data = data.reshape(h, w, c).astype(data_type)
        return data, label


class FFTCut(nn.Module):
    def __init__(self, mode: str, low_percent: float = 0., high_percent: float = 0.):
        super().__init__()
        mode = mode.lower()
        self.mode = mode
        self.low_percent = low_percent
        self.high_percent = high_percent
        assert self.mode in ['', 'l', 'h', 'b']  # l for low-cut, h for high-cut, b for both
        if self.mode == 'l':
            assert 0. <= self.low_percent <= 1
        elif self.mode == 'h':
            assert 0. <= self.high_percent <= 1.
        elif self.mode == 'b':
            assert 0. <= self.low_percent <= self.high_percent <= 1.

    def forward(self, data, labels):
        if self.mode == '':
            return data, labels
        _, h, w = data.size()
        img_f = torch.fft.fft2(data)
        # TODO： 好像反了
        if self.mode in ['h', 'b']:
            highest_band_height = int(h * (1. - self.high_percent))
            highest_band_width = int(w * (1. - self.high_percent))
            highest_band_h = highest_band_height // 2
            highest_band_w = highest_band_width // 2

            img_low = torch.empty(data.size(), dtype=img_f.dtype, device=img_f.device)
            # 低频在中心，被去除
            img_low[:, :highest_band_h, :highest_band_w] = img_f[:, :highest_band_h, :highest_band_w]  # upper left
            img_low[:, -highest_band_h:, :highest_band_w] = img_f[:, -highest_band_h:, :highest_band_w]  # lower left
            img_low[:, :highest_band_h, -highest_band_w:] = img_f[:, :highest_band_h, -highest_band_w:]  # upper right
            img_low[:, -highest_band_h:, -highest_band_w:] = img_f[:, -highest_band_h:, -highest_band_w:]  # lower right
            img_res = torch.fft.ifft2(img_low)
            img_res = torch.real(img_res)
            # print("Processing high cut")
        if self.mode in ['l', 'b']:
            lowest_band_height = int(h * self.low_percent)
            lowest_band_width = int(w * self.low_percent)
            lowest_band_h = lowest_band_height // 2
            lowest_band_w = lowest_band_width // 2

            if self.mode == 'l':
                img_high = img_f.clone()
            else:
                img_high = img_low.clone()
            # print(lowest_band_w)
            # 只保留中心的低频
            img_high[:, :lowest_band_h, :lowest_band_w] = 0
            img_high[:, -lowest_band_h:, :lowest_band_w] = 0
            img_high[:, :lowest_band_h, -lowest_band_w:] = 0
            img_high[:, -lowest_band_h:, -lowest_band_w:] = 0
            img_res = torch.fft.ifft2(img_high)
            img_res = torch.real(img_res)
            # print("Processing low cut")
        return img_res, labels
