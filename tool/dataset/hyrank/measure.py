import os
import scipy.io as sio

from collections import Counter

path = r'E:/zzy/GAN/data/HyRANK'
files = os.listdir(path)
for file in files:
    if 'gt' in file:
        gt = sio.loadmat(os.path.join(path, file))['map'].flatten()
        print(file, Counter(gt))

# TODO: Loukia的样本数与TSTNet中对不上
"""
Dioni_gt.mat Counter({0: 323976, 10: 6374, 9: 5035, 5: 1768, 11: 1754, 13: 1612, 1: 1262, 3: 614, 12: 492, 14: 398, 7: 361, 2: 204, 4: 150})
Dioni_gt_out68.mat Counter({0: 323976, 10: 6374, 9: 5035, 5: 1768, 11: 1754, 13: 1612, 1: 1262, 3: 614, 12: 492, 14: 398, 7: 361, 2: 204, 4: 150})
Loukia_gt.mat Counter({0: 221802, 9: 3793, 10: 2803, 5: 1401, 13: 1393, 8: 1072, 3: 542, 7: 500, 12: 487, 14: 451, 11: 404, 1: 288, 6: 223, 4: 79, 2: 67})
Loukia_gt_out68.mat Counter({0: 223097, 9: 3793, 10: 2803, 5: 1401, 13: 1393, 3: 542, 7: 500, 12: 487, 14: 451, 11: 404, 1: 288, 4: 79, 2: 67})
"""
