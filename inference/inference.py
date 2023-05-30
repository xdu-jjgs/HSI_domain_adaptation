import os
import torch
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from apex import amp
from tqdm import tqdm
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from models import build_model
from datas import build_dataset, build_dataloader
from plot import plot_confusion_matrix, plot_classification_image


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
    parser.add_argument('--opt-level',
                        type=str,
                        default='O0',
                        help='optimization level for nvidia/apex')
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
                            init_method=f'tcp://{args.master_ip}:{args.master_port}',
                            world_size=args.world_size,
                            rank=rank_process)
    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # build dataset
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = test_dataset.num_classes
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    # build data loader
    test_dataloader = build_dataloader(test_dataset, sampler=test_sampler, drop_last=False)
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)

    model = amp.initialize(model, opt_level=args.opt_level)
    model = DistributedDataParallel(model, broadcast_buffers=False)

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
    test_bar = tqdm(test_dataloader, desc='inferring', ascii=True)
    gts = []
    res = []
    with torch.no_grad():  # disable gradient back-propagation
        for batch, (x, label) in enumerate(test_bar):
            # change DoubleTensor x to FloatTensor
            x, label = x.float().to(device), label.to(device)
            _, y = model(x)

            pred = y.argmax(axis=1)

            res.append(pred.data.cpu().numpy())
            gts.append(label.data.cpu().numpy())
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
    res = np.concatenate(res)
    gts = np.concatenate(gts)
    PA, mPA, Ps, Rs, F1S = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
    logging.info('inference | PA={:.3f} mPA={:.3f}'.format(PA, mPA))
    for c in range(NUM_CLASSES):
        logging.info(
            'inference | class={}-{} P={:.3f} R={:.3f} F1={:.3f}'.format(c, test_dataset.names[c],
                                                                         Ps[c],
                                                                         Rs[c], F1S[c]))
    # logging.info(metric.matrix)
    plot_confusion_matrix(metric.matrix, os.path.join(args.path, 'confusion_matrix.png'))
    plot_classification_image(test_dataset, res, os.path.join(args.path, 'classification_map.png'))
    plot_classification_image(test_dataset, gts, os.path.join(args.path, 'gt_map.png'))


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
