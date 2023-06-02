import numpy as np
import matplotlib.pyplot as plt

from datas.base import HSIDataset


def plot_confusion_matrix(confusion_matrix, path, normalize: bool = True):
    if normalize:
        rows_sum = confusion_matrix.sum(axis=1)
        confusion_matrix = confusion_matrix / rows_sum[:, np.newaxis]
    num_classes = confusion_matrix.shape[0]
    tick_marks = range(0, num_classes)
    classes = [str(i + 1) for i in tick_marks]

    plt.figure(figsize=(16, 16))
    plt.xticks(tick_marks, classes, size=12)
    plt.yticks(tick_marks, classes, size=12)
    plt.xlabel('gt', fontsize=12)
    plt.ylabel('pred', fontsize=12)
    iters = np.reshape([[[i, j] for j in range(num_classes)] for i in range(num_classes)], (confusion_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(round(confusion_matrix[i, j], 2)), fontsize=12, va='center', ha='center')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.GnBu)  # 按照像素显示出矩阵
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')


def plot_classification_image(dataset: HSIDataset, pred, path):
    # print(dataset.gt_raw.shape)
    height, weight = dataset.gt_raw.shape
    image = np.zeros((height, weight, 3), dtype=np.uint8)
    coordinates = dataset.coordinates
    for ind, ele in zip(coordinates, pred):
        x, y = ind[0], ind[1]
        image[x][y] = dataset.pixels[ele]
    plt.imsave(path, image)
