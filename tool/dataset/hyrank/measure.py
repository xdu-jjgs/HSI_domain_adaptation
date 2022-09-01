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
        print(Counter(gt))

"""
train - Counter({8: 3793, 9: 2803, 4: 1401, 12: 1393, 7: 1072, 2: 542, 6: 500, 11: 487, 13: 451, 10: 404, 0: 288, 
5: 223, 3: 79, 1: 67})
test - Counter({9: 6374, 8: 5035, 4: 1768, 10: 1754, 12: 1612, 0: 1262, 2: 614, 11: 492, 
13: 398, 6: 361, 1: 204, 3: 150}) 
"""
