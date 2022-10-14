# from torch import argmax
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score


def one_hot(lable, class_number):
    '''转变标签形式'''
    one_hot_array = np.zeros([len(lable), class_number])
    for i in range(len(lable)):
        one_hot_array[i, lable[i]] = 1
    one_hot_array = one_hot_array.astype(np.float32)
    return one_hot_array
    # return torch.eye(class_number)[label, :]


def get_criteria(y_pred, y_real, class_num):
    # y_pred = torch.argmax(y_pred, dim=1)
    # y_pred = argmax(y_pred, dim=1)
    # y_real = torch.argmax(y_real, dim=1)
    y_pred = y_pred.cpu().numpy()
    y_real = y_real.cpu().numpy()
    oa = accuracy_score(y_real, y_pred)
    label_list = [i for i in range(class_num)]
    per_class_acc = recall_score(y_real, y_pred, labels=label_list, average=None, zero_division=0)
    aa = np.mean(per_class_acc)
    kappa = cohen_kappa_score(y_real, y_pred)
    return oa, aa, kappa, per_class_acc


