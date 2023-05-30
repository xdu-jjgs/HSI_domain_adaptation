import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from PIL import Image
from datetime import datetime
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from auxiliary_func import *
from models import build_model
from criterions import build_criterion
from datas import build_dataset, build_dataloader
from plotter import plot_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
                        default=None,
                        help='checkpoint file')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('../../runs', datetime.now().strftime('%Y%m%d-%H%M%S-train')),
                        help='path for experiment output files')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='whether not to validate in the training process')
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
    parser.add_argument('--opt-level',
                        type=str,
                        default='O0',
                        help='optimization level for nvidia/apex')
    args = parser.parse_args()
    # number of GPUs totally, which equals to the number of processes
    args.world_size = args.nodes * args.gpus
    return args


def worker(rank_gpu, args):
    # create experiment output path if not exists
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    assert CFG.EPOCHS % args.world_size == 0, 'cannot apportion epoch to gpus averagely'
    # log to file and stdout
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
    # number of workers
    logging.info('train on {} of {} processes'.format(rank_process + 1, dist.get_world_size()))

    # use device cuda:n in the process #n
    torch.cuda.set_device(rank_gpu)
    device = torch.device('cuda', rank_gpu)

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # initialize TensorBoard summary writer
    # rank of global worker
    # if dist.get_rank() == 0:
    #     writer = SummaryWriter(logdir=args.path)

    # build dataset
    # train_dataset = build_dataset('train')
    # test_dataset = build_dataset('test')
    val_dataset = build_dataset('val')
    # assert train_dataset.num_classes == val_dataset.num_classes
    logging.info("Number of val {}".format(len(val_dataset)))
    NUM_CHANNELS = val_dataset.num_channels
    NUM_CLASSES = val_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    # build data sampler
    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    # train_sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.get_labels())
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    # test_sampler = DistributedSampler(test_dataset, shuffle=True)
    # test_sampler = ImbalancedDatasetSampler(test_dataset, labels=test_dataset.get_labels())
    print('GT:', type(val_dataset.get_labels()))
    print('shape of GT:', val_dataset.get_labels().shape)
    # build data loader
    # train_dataloader = build_dataloader(train_dataset, sampler=train_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)
    # test_dataloader = build_dataloader(test_dataset, sampler=test_sampler)

    val_criterion = build_criterion(CFG.CRITERION.ITEMS4, CFG.CRITERION.WEIGHTS4)
    val_criterion.to(device)

    # build model TODO: notice here
    G, D, C = build_model(NUM_CHANNELS, NUM_CLASSES)
    G.to(device)
    D.to(device)
    C.to(device)

    # DDP
    G = DistributedDataParallel(G)
    D = DistributedDataParallel(D)
    C = DistributedDataParallel(C)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_OA = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        G.load_state_dict(checkpoint['G']['state_dict'])
        D.load_state_dict(checkpoint['D']['state_dict'])
        C.load_state_dict(checkpoint['C']['state_dict'])
        # epoch = checkpoint['optimizer']['epoch']
        # iteration = checkpoint['optimizer']['iteration']
        best_OA = checkpoint['metric']['OA']
        # best_epoch = checkpoint['optimizer']['best_epoch']
        logging.info('load checkpoint {} with OA={:.4f}, epoch={}'.format(args.checkpoint, best_OA, epoch))

    # train - validation loop

    # validate
    G.eval()  # set model to evaluation mode
    D.eval()  # set model to evaluation mode
    C.eval()  # set model to evaluation mode
    # TODO attention:如果 retain graph = true 此处不能用eval() 否则计算图会被free掉 导致模型失效
    # metric.reset()  # reset metric
    val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
    val_loss = 0.
    with torch.no_grad():  # disable gradient back-propagation
        for x_t, label, index in val_bar:
            x_t, label = x_t.to(device), label.to(device)
            y_t = C(x_t)

            loss = val_criterion(y_t, label)
            val_loss += loss.item()

            pred = y_t.argmax(axis=1)
            val_dataset.update_pred(index.cpu(), pred.cpu())
            val_dataset.update_gtmap(index.cpu(), label.cpu())
            oa, aa, kappa, per_class_acc = get_criteria(pred.cpu().numpy(), label.cpu().numpy(), NUM_CLASSES)

            val_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{loss.item():.4f}',
                'OA': f'{oa:.4f}',
                'AA': f'{aa:.4f}',
                'Kappa': f'{kappa:.4f}'
            })

    # plot
    pred_map = val_dataset.get_pred()    # [h,w]
    gt_map = val_dataset.get_gtmap()
    plotter = plot_label(CFG.DATASET.NAME, NUM_CLASSES)
    img = plotter.plot_color(pred_map)
    ret_im = Image.fromarray(np.uint8(img)).convert('RGB')
    ret_im.save(os.path.join(args.path, 'CCGDA_source.tif'))
    img = plotter.plot_color(gt_map)
    ret_im = Image.fromarray(np.uint8(img)).convert('RGB')
    ret_im.save(os.path.join(args.path, 'gt_source.tif'))



def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
