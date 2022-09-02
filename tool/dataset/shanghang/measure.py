import os
import numpy as np

from collections import Counter

path = r'E:\zts\dataset\shanghaihangzhou_preprocessed'
files = os.listdir(path)
for file in files:
    if file.split('.')[-1] == 'npy':
        if '_gt'in file:
            gt = np.load(os.path.join(path, file))
            print(file, gt.shape, Counter(gt))

'''
test_gt.npy (368000,) Counter({1: 161689, 0: 123123, 2: 83188})
train_gt.npy (135700,) Counter({1: 77450, 2: 40207, 0: 18043})
'''