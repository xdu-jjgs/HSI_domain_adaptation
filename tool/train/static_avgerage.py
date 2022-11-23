import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs/old'
datasets = ['houston', 'hyrank', 'shanghang']
# datasets = ['houston', 'houston_sample']
pattern = re.compile(r'Best epoch:\d+, PA:(0.\d+)')

for dataset in datasets:
    path = os.path.join(root, dataset)
    models_paths = [os.path.join(path, i) for i in os.listdir(path)]
    for mp in models_paths:
        log_file_path = os.path.join(mp, 'train.log')
        with open(log_file_path) as fo:
            model = os.path.basename(mp).split('-')[0]
            text = fo.read()
            try:
                pas = list(map(lambda x: float(x), re.findall(pattern, text)))
                ave = sum(pas) / len(pas)
                min_ = min(pas)
                max_ = max(pas)
                print(dataset, model, pas, "ave:{}, min:{}, max:{}".format(ave, min_, max_))
            except ZeroDivisionError:
                print("No result in dataset {}, model {}".format(dataset, model))
