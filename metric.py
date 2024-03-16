import numpy as np


class Metric:
    def __init__(self, num_classes, ignore_indexes=[]):
        self.num_classes = num_classes
        self.ignore_indexes = ignore_indexes
        self.matrix = np.zeros((self.num_classes, self.num_classes))
        self.count = 0

    def reset(self):
        self.matrix.fill(0)
        self.count = 0

    def add(self, pred, label):
        mask = (label >= 0) & (label < self.num_classes)
        for ignore_index in self.ignore_indexes:
            mask &= (label != ignore_index)

        count = np.bincount(self.num_classes * label[mask] + pred[mask], minlength=self.num_classes ** 2)
        self.matrix += count.reshape((self.num_classes, self.num_classes))
        if len(pred.shape) >= 1:
            self.count += pred.shape[0]
        else:
            self.count += 1

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
        with np.errstate(divide='ignore', invalid='ignore'):
            accs = np.diag(self.matrix) / self.matrix.sum(axis=0)
            accs = np.nan_to_num(accs)  # 将 NaN 替换为 0
        return accs

    def Rs(self):
        # recall of each class
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = np.diag(self.matrix) / self.matrix.sum(axis=1)
            rs = np.nan_to_num(rs)  # 沿着1求和，相当于excel求到最右侧的单元格
        return rs

    def F1s(self):
        # F1 score
        ps = self.Ps()
        rs = self.Ps()
        with np.errstate(divide='ignore', invalid='ignore'):
            f1s = 2 * ps * rs / (ps + rs)
            f1s = np.nan_to_num(f1s)  # 将 NaN 替换为 0
        return f1s

    def KC(self):
        # Kappa Coefficient
        p0 = self.PA()
        pe = np.sum(np.diag(self.matrix) * self.matrix.sum(axis=1)) / np.sum(self.matrix)**2
        kc = (p0 - pe) / (1 - pe)
        return kc

