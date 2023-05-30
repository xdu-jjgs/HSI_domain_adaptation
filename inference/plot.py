import cv2
import numpy as np
import matplotlib.pyplot as plt

from datas.base import HSIDataset


def plot_confusion_matrix(confusion_matrix, path):
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
        plt.text(j, i, format(confusion_matrix[i, j]), fontsize=12, va='center', ha='center')

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.GnBu)  # 按照像素显示出矩阵
    plt.colorbar()
    plt.savefig(path, bbox_inches='tight')

def plot_classification_image(dataset: HSIDataset, pred, path):
    height, weight = dataset.gt.shape
    image = np.zeros((height, weight, 3))
    coordinates = dataset.coordinates
    for ind, ele in zip(coordinates, pred):
        image[ind] = dataset.pixels[ele]
    cv2.imsave(image, path)