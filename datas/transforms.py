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


class CropImage(nn.Module):
    # 整体
    def __init__(self, window_size: Tuple[int, int], pad_mode: str, selector=None, return_type: str = 'data'):
        super(CropImage, self).__init__()
        assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1, 'window size should be odd!'
        assert return_type in ['data', 'coordinate']
        self.window_size = window_size
        self.pad_mode = pad_mode
        self.selector = selector
        self.return_type = return_type

    def forward(self, image, label):
        h, w, c = image.shape

        patch = ((self.window_size[0] - 1) // 2, (self.window_size[1] - 1) // 2)
        image = np.pad(image, (patch, patch, (0, 0)), self.pad_mode)
        images = []
        labels = []
        coordinates = []
        for i in range(h):
            for j in range(w):
                if self.selector is None or self.selector(image[i, j], label[i, j]):
                    images.append(image[i:i + self.window_size[0], j:j + self.window_size[1], ...])
                    labels.append(label[i][j])
                    coordinates.append([i, j])
        if self.return_type == 'data':
            images = np.stack(images, axis=0)
            labels = np.array(labels)
            return images, labels
        else:
            coordinates = np.array(coordinates)
            return coordinates, labels


class DataAugment(nn.Module):
    # 整体
    def __init__(self, ratio: int = 1, trans=None):
        super(DataAugment, self).__init__()
        self.ratio = ratio
        self.trans = trans

    def forward(self, image, label):
        shape = image.size()
        print(shape)
        assert len(shape) == 4
        # NCHW
        image_concat = torch.cat([image for _ in range(self.ratio)], dim=0)
        label_concat = np.concatenate([label for _ in range(self.ratio)])

        # TODO: Add trans for augmentation
        if self.trans:
            print("Augment with {}".format(self.trans))
        print(image_concat.size())
        return image_concat, label_concat
