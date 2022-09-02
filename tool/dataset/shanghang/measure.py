import os
import torch
import numpy as np

from collections import Counter

path = r'E:\zts\dataset\shanghaihangzhou_preprocessed'
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
