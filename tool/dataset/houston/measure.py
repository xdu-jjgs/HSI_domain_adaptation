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
train - Counter({7: 443, 6: 408, 3: 365, 2: 365, 1: 345, 5: 319, 4: 285})
test - Counter({6: 32459, 7: 6365, 5: 5347, 2: 4888, 3: 2766, 1: 1353, 4: 22})
"""