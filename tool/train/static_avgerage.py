import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs/'
datasets = ['houston', 'houston_sample', 'hyrank', 'hyrank_sample', 'shanghang', 'shanghang_sample']
pattern = re.compile(r'Best([\s\w]*)\sepoch:\d+, PA:(0.\d+)')

for dataset in datasets:
    path = os.path.join(root, dataset)
    models_paths = [os.path.join(path, i) for i in os.listdir(path)]
    for mp in models_paths:
        log_file_path = os.path.join(mp, 'train.log')
        model = os.path.basename(mp).split('-')[0]
        try:
            with open(log_file_path) as fo:
                text = fo.read()
            res = {}
            pair = re.findall(pattern, text)
            sub_model = list(map(lambda x: x[0].replace(' ', '_'), pair))
            pas = list(map(lambda x: float(x[1]), pair))
            for ele, pa in zip(sub_model, pas):
                if res.get(model+ele):
                    res[model+ele].append(pa)
                else:
                    res[model + ele] = [pa]
            for name, v in res.items():
                ave = sum(v) / len(v)
                min_ = min(v)
                max_ = max(v)
                print(dataset, name, "times:{}, ave:{:.3f}, +{:.3f} -{:.3f}".
                      format(len(v), ave, max_-ave, ave-min_, ))
        except (ZeroDivisionError, FileNotFoundError):
            print("No result in dataset {}, model {}".format(dataset, model))
