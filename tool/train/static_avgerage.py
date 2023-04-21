import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs/'
datasets = ['houston', 'houston_sample', 'hyrank', 'hyrank_sample', 'shanghang', 'shanghang_sample']
pattern = re.compile(r'Best([\s\w]*)\sepoch:\d+, PA:(0.\d+)')


for dataset in datasets:
    path = os.path.join(root, dataset)
    models_paths = [os.path.join(path, i) for i in os.listdir(path)]
    for mp in models_paths:
        model = os.path.basename(mp).split('-')[0]
        seed_path = [os.path.join(path, i) for i in os.listdir(path)]
        for sp in seed_path:
            log_file_path = os.path.join(sp, 'train.log')
            res = {}
            try:
                with open(log_file_path) as fo:
                    text = fo.read()
                    pair = re.findall(pattern, text)[0]
                    sub_model = pair[0].replace(' ', '_')
                    pa = float(pair[1])
                    if res.get(model+sub_model):
                        res[model+sub_model].append(pa)
                    else:
                        res[model + sub_model] = [pa]
            except (ZeroDivisionError, FileNotFoundError):
                print("No result in dataset {}, model {}".format(dataset, model))

            for name, v in res.items():
                v.sort()
                avg = sum(v) / len(v)
                min_ = v[0]
                max_ = v[-1]
                print(dataset, name, "times:{}, avg:{:.3f}±{:.3f} best:{:.3f} med:{:.3f}".
                      format(len(v), avg, (max_ - min_)/2, max_, v[(len(v)-1)//2]))

