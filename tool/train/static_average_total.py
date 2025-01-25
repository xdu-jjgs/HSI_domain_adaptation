import os
import re
import numpy as np

root = r'E:/zts/HSI_domain_adaptation/runs'
# datasets = ['houston', 'hyrank', 'shanghang']
datasets = ['houston_hyper2', 'hyrank_hyper2', 'shanghang_hyper2']
pattern_best_pa = re.compile(r'Best epoch:(\d+), PA:(0.\d+)')

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
                    pair = re.findall(pattern_best_pa, text)
                    try:
                        if seed is not None:
                            # assert len(pair) == 1, 'Error in {}'.format(log_path)
                            pair = [pair[-1]]
                        epochs = list(map(lambda x: x[0], pair))
                        pas = list(map(lambda x: x[1], pair))
                        for epoch, pa in zip(epochs, pas):
                            others = re.findall(r'val epoch=\d+ \| PA={} mPA=[.na\d]+ KC=(0.\d+)[\w\W]+?'
                                                r'(class=[\w\W]+?)(train|Best)'.format(pa), text)[0]
                            kappa = others[0]
                            class_pas = re.findall(r'P=([01].\d+)', others[1].replace('nan', '0.000'))
                            class_pas = list(map(lambda x: float(x), class_pas))

                            if res.get(model_name):
                                res[model_name][0].append(float(pa))
                                res[model_name][1].append(int(epoch))
                                res[model_name][2].append(float(kappa))
                                res[model_name][3].append(sum(class_pas) / len(class_pas))
                                res[model_name][4].append(class_pas)
                            else:
                                res[model_name] = [[float(pa)], [int(epoch)], [float(kappa)],
                                                   [sum(class_pas) / len(class_pas)], [class_pas]]
                    except IndexError:
                        print("No result in {}".format(log_path))

    for name, v in res.items():
        pas, epochs, kappas, aas, class_pas = v
        avg_epoch = sum(epochs) / len(epochs)
        pas = list(map(lambda x: x * 100, pas))
        pas.sort()
        avg_pa = sum(pas) / len(pas)
        min_pa, max_pa = pas[0], pas[-1]

        kappas = list(map(lambda x: x * 100, kappas))
        kappas.sort()
        avg_kappa = sum(kappas) / len(kappas)
        min_kappa, max_kappa = kappas[0], kappas[-1]

        aas = list(map(lambda x: x * 100, aas))
        aas.sort()
        avg_aa = sum(aas) / len(aas)
        min_aa, max_aa = aas[0], aas[-1]

        class_pas = np.array(class_pas) * 100

        avg_class_pas = np.mean(class_pas, axis=0)
        max_class_pas = np.max(class_pas, axis=0)
        min_class_pas = np.min(class_pas, axis=0)
        class_pas_out = ['{:.1f}±{:.1f}'.format(i, j) for i, j in zip(avg_class_pas, np.std(class_pas, axis=0))]
        print(dataset, name, "times:{}, avg:{:.1f}±{:.1f} best:{:.1f} med:{:.1f} "
                             "aa:{:.1f}±{:.1f} kappa:{:.1f}±{:.1f} epoch_avg:{:.1f} max:{}, name:{} class_pas:{}"
              .format(len(pas), avg_pa, np.std(pas), max_pa, pas[(len(pas) - 1) // 2],
                      avg_aa, np.std(aas),
                      avg_kappa, np.std(kappas),
                      avg_epoch, max(epochs),
                      name, class_pas_out
                      ))

    # print(dataset, res)
