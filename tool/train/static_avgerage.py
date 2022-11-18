import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs'
dirs = ['houston', 'hyrank', 'shanghang']

for d in dirs:
    path = os.path.join(root, d)
    models_path = [os.path.join(path, i) for i in os.listdir(path)]
