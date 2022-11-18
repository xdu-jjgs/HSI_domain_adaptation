import os
import scipy.io as sio

from collections import Counter

file = r'E:/zzy/GAN/data/ShanghaiHangzhou/DataCube_ShanghaiHangzhou.mat'
gt1 = sio.loadmat(file)['gt1'].flatten()
print('gt1', Counter(gt1))
gt2 = sio.loadmat(file)['gt2'].flatten()
print('gt2', Counter(gt2))

'''
gt1 Counter({2: 161689, 1: 123123, 3: 83188})
gt2 Counter({2: 77450, 3: 40207, 1: 18043})
'''