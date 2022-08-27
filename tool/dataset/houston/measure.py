import os
import torch
import numpy as np

from collections import Counter

path = r'E:\zts\dataset\houston_preprocessed'
files = os.listdir(path)
for file in files:
    if file.split('.')[-1] == 'pt':
        data = torch.load(os.path.join(path, file))
        # NCHW -> CHWN
        data = torch.permute(data, (1,2,3,0))
        c = data.size()[0]
        data = torch.reshape(data, (c, -1))
        print(torch.mean(data, dim=-1))
        print(torch.std(data, dim=-1))
    elif file.split('.')[-1] == 'npy':
        gt = np.load(os.path.join(path, file))
        print(Counter(gt))

"""
train - Counter({6: 443, 5: 408, 2: 365, 1: 365, 0: 345, 4: 319, 3: 285})
test - Counter({5: 32459, 6: 6365, 4: 5347, 1: 4888, 2: 2766, 0: 1353, 3: 22})
"""