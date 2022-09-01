import os
import torch
import numpy as np

from collections import Counter

path = r'E:\zts\dataset\hyrank_preprocessed'
files = os.listdir(path)
for file in files:
    if file.split('.')[-1] == 'pt':
        data = torch.load(os.path.join(path, file)).float()
        # NCHW -> CHWN
        data = torch.permute(data, (1, 2, 3, 0))
        c = data.size()[0]
        data = torch.reshape(data, (c, -1))
        print(torch.mean(data, dim=-1))
        print(torch.std(data, dim=-1))
    elif file.split('.')[-1] == 'npy':
        gt = np.load(os.path.join(path, file))
        print(gt.shape, Counter(gt))

# TODO: Loukia的样本数与TSTNet中对不上
"""
test - (12208,) Counter({6: 3793, 7: 2803, 4: 1401, 10: 1393, 2: 542, 5: 500, 9: 487, 11: 451, 8: 404, 0: 288, 
3: 79, 1: 67}) 
train - (20024,) Counter({7: 6374, 6: 5035, 4: 1768, 8: 1754, 10: 1612, 0: 1262, 2: 614, 9: 492, 11: 398, 5: 361, 
1: 204, 3: 150})
"""
