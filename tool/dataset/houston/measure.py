import os
import scipy.io as sio

from collections import Counter

path = r'E:/zzy/GAN/data/Houston'
files = os.listdir(path)
for file in files:
    if 'gt' in file:
        gt = sio.loadmat(os.path.join(path, file))['map'].flatten()
        print(file, Counter(gt))

"""
Houston13_7gt.mat Counter({0: 197810, 7: 443, 6: 408, 3: 365, 2: 365, 1: 345, 5: 319, 4: 285})
Houston18_7gt.mat Counter({0: 147140, 6: 32459, 7: 6365, 5: 5347, 2: 4888, 3: 2766, 1: 1353, 4: 22})
"""
