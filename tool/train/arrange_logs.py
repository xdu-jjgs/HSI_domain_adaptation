import os

path = r'E:\zts\HSI_domain_adaptation\runs\shanghang'
models_paths = [os.path.join(path, i) for i in os.listdir(path)]
for mp in models_paths:
    log_files = list(filter(lambda x: 'events' in x, os.listdir(mp)))
    # log_files = [os.path.join(mp, i) for i in log_files]
    for ind, ele in enumerate(log_files):
        dir_file = os.path.join(mp, str(ind + 1))
        if not os.path.exists(dir_file):
            os.mkdir(dir_file)
        os.rename(os.path.join(mp, ele), os.path.join(dir_file, ele))
