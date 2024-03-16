import os
import re
import matplotlib.pyplot as plt
import numpy as np

path = [
    # r'runs/hyrank/nommd-train/1',
    # r'runs/hyrank/ddc-train/1',
    r'runs/hyrank/dan-train/1',
    r'runs/hyrank/jan-train/1',
    # r'runs/hyrank/dsan-train/1',
    r'runs/hyrank/dann-train/1',
    # r'runs/hyrank/mcd-train/1',
    # r'runs/hyrank/dst_1_1_1_07_2-train/1'
]

plt.xlabel('Magnitude of Standard Deviations')
plt.ylabel('Number of Channels')

ans = []
for p in path:
    files = os.listdir(p)
    for f in files:
        res = re.findall(r'std_mix_(\w+)_(\d+)', f)
        if res:
            model_name, pa = res[0]
            file_path = os.path.join(p, f)
            data = np.load(file_path)
            # data = data[np.random.choice(len(data), 100)]
            ans.append([model_name, pa, np.mean(data)])
            print(model_name, len(data))
            plt.hist(data, bins=20, alpha=0.5, label='{}'.format(model_name.upper(), pa))
ans.sort(key=lambda x: x[1], reverse=True)
print(*ans)
# plt.xlim(-0.2, 6)
plt.legend()
plt.show()
