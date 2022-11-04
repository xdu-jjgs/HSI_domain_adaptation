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
