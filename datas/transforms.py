import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from typing import List, Tuple


class Compose(nn.Module):
    def __init__(self, transforms: List[nn.Module]):
        super(Compose, self).__init__()
        self.transforms = transforms

    def forward(self, image, label):
        for transform in self.transforms:
            if image is not None:
                image, label = transform(image, label)
        return image, label


class RandomCrop(nn.Module):
    # 元素
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.size = size

    def forward(self, image, label):
        if image is not None:
            h, w, _ = image.shape
            new_h, new_w = self.size

            top = np.random.randint(0, h - new_h)
            down = top + new_h
            left = np.random.randint(0, w - new_w)
            right = left + new_w

            if image is not None:
                image = image[top:down, left:right, :]
            if label is not None:
                label = label[top:down, left:right]

        return image, label


class ToTensor(nn.Module):
    # 元素/整体
    def forward(self, image, label):
        shape = image.shape
        image = torch.tensor(image, dtype=torch.float32)
        if len(shape) == 4:
            image = image.permute((0, 3, 1, 2))
        elif len(shape) == 3:
            image = image.permute((2, 0, 1))
        return image, label


class ToTensorPreData(nn.Module):
    # 整体
    def __init__(self):
        super(ToTensorPreData, self).__init__()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image, label):
        image = [self.to_tensor(data) for data in image]
        return image, label


class ToTensorPreSubData(nn.Module):
    # 整体
    def forward(self, image, label):
        image = [torch.tensor(data) for data in image]
        image = [data.permute((0, 3, 1, 2)) for data in image]
        return image, label


class Normalize(nn.Module):
    # 整体/元素
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.normalize = transforms.Normalize(mean, std)

    def forward(self, image, label):
        image = self.normalize(image)
        return image, label


class LabelRenumber(nn.Module):
    # 整体/元素
    def forward(self, image, label):
        def renumber(ele):
            return label_cur.index(ele)

        renumber_np = np.frompyfunc(renumber, 1, 1)
        label_cur = list(np.unique(label))
        label = renumber_np(label).astype('int')

        return image, label


class ZScoreNormalize(nn.Module):
    # 整体
    def forward(self, image, label):
        h, w, c = image.shape
        data_type = image.dtype
        image = image.reshape(h * w, c).astype('float32')
        mean = np.mean(image, axis=0)
        # std上溢了，需要用float32
        std = np.std(image, axis=0)
        # print("mean:{}, std: {}".format(mean, std))
        image = (image - mean) / std
        image = image.reshape(h, w, c).astype(data_type)
        return image, label

