import os
import re
import matplotlib.pyplot as plt
import numpy as np


path = [
    r'runs/houston/nommd-train/1',
    r'runs/houston/ddc-train/1',
    r'runs/houston/dan-train/1',
    r'runs/houston/jan-train/1',
    r'runs/houston/dsan-train/1',
    r'runs/houston/dann-train/2',
    r'runs/houston/mcd-train/1',
    r'runs/houston/dst_1_1_1_07_2-train/1'
]

plt.xlabel('Magnitude of Standard Deviations')
plt.ylabel('Number of Channels')

for p in path[2:]:
    files = os.listdir(p)
    for f in files:
        res = re.findall(r'std_mix_(\w+)_(\d+)', f)
        if res:
            model_name, pa = res[0]
            print(model_name, pa)
            file_path = os.path.join(p, f)
            plt.hist(np.load(file_path), bins=30, alpha=0.5, label='{}_{}'.format(model_name, pa))

# plt.xlim(-0.5,2)
plt.legend()
plt.show()

