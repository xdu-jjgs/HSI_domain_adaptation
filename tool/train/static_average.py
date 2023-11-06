import os
import re

root = r'E:/zts/HSI_domain_adaptation/runs'
datasets = ['houston', 'houston_sample', 'hyrank', 'hyrank_sample', 'shanghang', 'shanghang_sample', 'pavia', 'indiana']
pattern = re.compile(r'Best epoch:(\d+), PA:(0.\d+)')

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
                        epochs = list(map(lambda x: int(x[0]), pair))
                        pas = list(map(lambda x: float(x[1]), pair))
                        for epoch, pa in zip(epochs,pas):
                            if res.get(model_name):
                                res[model_name][0].append(pa)
                                res[model_name][1].append(epoch)
                            else:
                                res[model_name] = [[pa], [epoch]]
                    except IndexError:
                        print("No result in {}".format(log_path))

    for name, v in res.items():
        pas, epochs = v
        avg_epoch = sum(epochs) / len(epochs)
        pas.sort()
        avg = sum(pas) / len(pas)
        min_ = pas[0]
        max_ = pas[-1]
        print(dataset, name, "times:{}, avg:{:.3f}±{:.3f} best:{:.3f} med:{:.3f} epoch_avg:{:.3f} max:{}".
              format(len(pas), avg, (max_ - min_) / 2, max_, pas[(len(pas) - 1) // 2], avg_epoch, max(epochs)))

    # print(dataset, res)
