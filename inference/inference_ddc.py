import os
import torch
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from datetime import datetime
from sklearn.manifold import TSNE
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from models import build_model
from datas import build_dataset, build_dataloader
from models.utils.stats import count_params, count_flops
from plot import plot_confusion_matrix, plot_classification_image, plot_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('checkpoint',
                        type=str,
                        help='checkpoint file')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    parser.add_argument('-n',
                        '--nodes',
                        type=int,
                        default=1,
                        help='number of nodes / machines')
    parser.add_argument('-g',
                        '--gpus',
                        type=int,
                        default=1,
                        help='number of GPUs per node / machine')
    parser.add_argument('-r',
                        '--rank-node',
                        type=int,
                        default=0,
                        help='ranking of the current node / machine')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='backend for PyTorch DDP')
    parser.add_argument('--master-ip',
                        type=str,
                        default='localhost',
                        help='network IP of the master node / machine')
    parser.add_argument('--master-port',
                        type=str,
                        default='8888',
                        help='network port of the master process on the master node / machine')
    parser.add_argument('--seed',
                        type=int,
                        default=30,
                        help='random seed')
    parser.add_argument('--sample-number',
                        type=int,
                        default=20,
                        help='random seed')
    args = parser.parse_args()
    # number of GPUs totally, which equals to the number of processes
    args.world_size = args.nodes * args.gpus
    return args


def worker(rank_gpu, args):
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    assert CFG.EPOCHS % args.world_size == 0, 'cannot apportion epoch to gpus averagely'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'test.log')),
            logging.StreamHandler(),
        ])

    # rank of global worker
    rank_process = args.gpus * args.rank_node + rank_gpu
    dist.init_process_group(backend=args.backend,
                            init_method=f'tcp://{args.master_ip}:{args.master_port}?use_libuv=False',
                            world_size=args.world_size,
                            rank=rank_process)
    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build dataset
    source_dataset = build_dataset('train')
    target_dataset = build_dataset('test')
    assert source_dataset.num_classes == target_dataset.num_classes
    logging.info(
        "Number of train {}, val {}, test {}".format(len(source_dataset), len(target_dataset), len(target_dataset)))

    NUM_CHANNELS = target_dataset.num_channels
    NUM_CLASSES = target_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    logging.info("Number of channels: {}".format(NUM_CHANNELS))
    source_sampler = DistributedSampler(source_dataset, shuffle=False)
    target_sampler = DistributedSampler(target_dataset, shuffle=False)
    # build data loader
    source_dataloader = build_dataloader(source_dataset, sampler=source_sampler, drop_last=False)
    target_dataloader = build_dataloader(target_dataset, sampler=target_sampler, drop_last=False)
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    print("Num of params of {} ({}) is {:.2f}M".format(CFG.MODEL.NAME, CFG.MODEL.BACKBONE, count_params(model)))
    model = DistributedDataParallel(model, broadcast_buffers=False)
    # build metric
    metric = Metric(NUM_CLASSES)

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    # delete module saved in train
    model.load_state_dict(checkpoint['model']['state_dict'])
    best_PA = checkpoint['metric']['PA']
    logging.info('load checkpoint {} with PA={:.3f}'.format(args.checkpoint, best_PA))

    # inference
    model.eval()  # set model to evaluation mode
    metric.reset()  # reset metric
    features_s, features_t = [], []
    labels_s, labels_t = [], []
    res = []
    with torch.no_grad():  # disable gradient back-propagation
        target_bar = tqdm(target_dataloader, desc='inferring-t', ascii=True)
        for batch, (x, label) in enumerate(target_bar):
            x, label = x.to(device), label.to(device)
            print(x.size())
            print("Num of FLOPs of {} ({}) is {:.2f}M".format(CFG.MODEL.NAME, CFG.MODEL.BACKBONE,
                                                              count_flops(model, x)))
            # raise NotImplementedError
            with autocast():
                f, _, y = model(x)
            pred = y.argmax(axis=1)
            f = torch.squeeze(f)
            features_t.append(f.data.cpu().numpy())
            res.append(pred.data.cpu().numpy())
            labels_t.append(label.data.cpu().numpy())
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
        source_bar = tqdm(source_dataloader, desc='inferring-s', ascii=True)
        for batch, (x, label) in enumerate(source_bar):
            x, label = x.to(device), label.to(device)
            with autocast():
                f, _, _ = model(x)
            f = torch.squeeze(f)
            features_s.append(f.data.cpu().numpy())
            labels_s.append(label.data.cpu().numpy())
    res = np.concatenate(res)
    labels_s, labels_t = np.concatenate(labels_s), np.concatenate(labels_t)
    features_s, features_t = np.concatenate(features_s), np.concatenate(features_t)
    PA, mPA, Ps, Rs, F1S = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
    logging.info('inference | PA={:.3f} mPA={:.3f}'.format(PA, mPA))
    for c in range(NUM_CLASSES):
        logging.info(
            'inference | class={}-{} P={:.3f} R={:.3f} F1={:.3f}'.format(c, target_dataset.names[c],
                                                                         Ps[c], Rs[c], F1S[c]))
    # std_s = np.std(features_s, axis=0)
    # std_t = np.std(features_t, axis=0)
    # print(features_s.shape)
    # print(features_t.shape)
    # features_mix = np.concatenate([features_s[:1000],
    #                                features_t[:1000]])
    # # features_mix = (features_mix - np.mean(features_mix)) / np.std(features_mix)
    # std_mix = np.std(features_mix, axis=0)
    # std_mix = std_mix[~np.isinf(std_mix)]
    #
    # print(std_mix.shape, np.min(std_mix), np.max(std_mix))
    # print(args.path, best_PA, np.mean(std_mix))
    # model_name = os.path.dirname(args.path).split('/')[-1].split('-')[0]
    # np.save(os.path.join(args.path, 'std_mix_{}_{}.npy'.format(model_name, int(best_PA * 1000))), std_mix)
    # raise NotImplementedError

    # plt.xlabel('Magnitude of Standard Deviations')
    # plt.ylabel('Number of Channels')
    # plt.hist(std_s, bins=20, alpha=0.5, label='Source Features')
    # plt.hist(std_t, bins=20, alpha=0.5, label='Target Features')
    # plt.hist(std_mix, bins=20, alpha=0.5, label='Mixture Features')
    # plt.legend()
    # plt.show()

    for i in range(NUM_CLASSES):
        fs = features_s[np.where(labels_s == i)]
        ft = features_t[np.where(labels_t == i)]
        np.random.shuffle(fs)
        np.random.shuffle(ft)
        fs, ft = fs[:args.sample_number], ft[:args.sample_number]
        if len(fs) < args.sample_number:
            print("Not enough samples for class{} in source domain".format(i))
        if len(ft) < args.sample_number:
            print("Not enough samples for class{} in target domain".format(i))
        num_fs = len(fs)
        f = np.concatenate([fs, ft], axis=0)
        tsne = TSNE(n_components=2)
        f = tsne.fit_transform(f)
        fs, ft = f[:num_fs], f[num_fs:]
        # fs = tsne.fit_transform(fs)
        # ft = tsne.fit_transform(ft)
        plot_features(fs, ft, os.path.join(args.path, 'feature_map_class{}.png'.format(i+1)))
    plot_confusion_matrix(metric.matrix, os.path.join(args.path, 'confusion_matrix.png'))
    plot_classification_image(target_dataset, res, os.path.join(args.path, 'classification_map.png'))
    plot_classification_image(target_dataset, labels_t, os.path.join(args.path, 'gt_map.png'))


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
