import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs/result_patch27_fe'
datasets = ['houston', 'houston_sample', 'hyrank', 'hyrank_sample', 'shanghang', 'shanghang_sample']
pattern = re.compile(r'Best([\s\w]*)\sepoch:\d+, PA:(0.\d+)')

for dataset in datasets:
    path = os.path.join(root, dataset)
    res = {}
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if 'train.log' == filename:
                # print(dirpath)
                if '-train' in os.path.basename(dirpath):
                    model_name = os.path.basename(dirpath)
                    seed = None
                else:
                    model_name, seed = os.path.basename(os.path.dirname(dirpath)), os.path.basename(dirpath)
                model_name = model_name.split('-')[0]
                log_path = os.path.join(dirpath, filename)

                with open(log_path) as fo:
                    text = fo.read()
                    pair = re.findall(pattern, text)
                    try:
                        if seed is not None:
                            pair = [pair[0]]
                        pas = list(map(lambda x: float(x[1]), pair))
                        for pa in pas:
                            if res.get(model_name):
                                res[model_name].append(pa)
                            else:
                                res[model_name] = [pa]
                    except IndexError:
                        print("No result in {}".format(log_path))

    for name, v in res.items():
        v.sort()
        avg = sum(v) / len(v)
        min_ = v[0]
        max_ = v[-1]
        print(dataset, name, "times:{}, avg:{:.3f}±{:.3f} best:{:.3f} med:{:.3f}".
              format(len(v), avg, (max_ - min_) / 2, max_, v[(len(v) - 1) // 2]))

    # print(dataset, res)
