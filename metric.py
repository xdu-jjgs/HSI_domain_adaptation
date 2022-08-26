import numpy as np


class Metric:
    def __init__(self, num_classes, ignore_indexes=[]):
        self.num_classes = num_classes
        self.ignore_indexes = ignore_indexes
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def reset(self):
        self.matrix.fill(0)

    def add(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        for ignore_index in self.ignore_indexes:
            mask &= (label != ignore_index)

        count = np.bincount(self.num_classes * label[mask] + pred[mask], minlength=self.num_classes ** 2)
        self.matrix += count.reshape((self.num_classes, self.num_classes))

    def PA(self):
        # pixel accuracy
        acc = np.diag(self.matrix).sum() / self.matrix.sum()
        return acc

    def mPA(self):
        # mean PA
        accs = self.Ps()
        acc = accs.mean()
        return acc

    def Ps(self):
        # precision of each class
        accs = np.diag(self.matrix) / self.matrix.sum(axis=0)
        return accs

    def Rs(self):
        # recall of each class
        rs = np.diag(self.matrix) / self.matrix.sum(axis=1)
        return rs

    def F1s(self):
        # F1 score
        ps = self.Ps()
        rs = self.Ps()
        f1s = 2 * ps * rs / (ps + rs)
        return f1s
