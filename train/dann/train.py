import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from apex import amp
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
from apex.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from auxiliary_func import *
from models import build_model
from criterions import build_criterion
from optimizers import build_optimizer
from schedulers import build_scheduler
from datas import build_dataset, build_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
    parser.add_argument('--checkpoint',
                        type=str,
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

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())
    # print(CFG)
    assert CFG.EPOCHS % args.world_size == 0, 'cannot apportion epoch to gpus averagely'
    # log to file and stdout
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'train.log')),
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
    if dist.get_rank() == 0:
        writer = SummaryWriter(logdir=args.path)

    # build dataset
    train_dataset = build_dataset('train')
    test_dataset = build_dataset('test')
    val_dataset = test_dataset
    assert train_dataset.num_classes == val_dataset.num_classes
    logging.info("Number of train {}, val {}, test {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    NUM_CHANNELS = train_dataset.num_channels
    NUM_CLASSES = train_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    # build data sampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    # build data loader
    train_dataloader = build_dataloader(train_dataset, sampler=train_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)
    test_dataloader = build_dataloader(test_dataset, sampler=test_sampler)

    # build model TODO: notice here
    DANN = build_model(NUM_CHANNELS, NUM_CLASSES)
    DANN.to(device)

    # print(model)
    # build criterion TODO: notice here
    train_criterion1 = build_criterion(CFG.CRITERION.ITEMS1, CFG.CRITERION.WEIGHTS1)
    train_criterion1.to(device)
    val_criterion = build_criterion(CFG.CRITERION.ITEMS4, CFG.CRITERION.WEIGHTS4)
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(DANN, CFG.OPTIMIZER.NAME1, CFG.OPTIMIZER.LR1)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    # mixed precision
    DANN, optimize = amp.initialize(DANN, optimizer, opt_level=args.opt_level)
    # DDP
    DANN = DistributedDataParallel(DANN)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_OA = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        DANN.load_state_dict(checkpoint['DANN']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_OA = checkpoint['metric']['OA']
        best_epoch = checkpoint['optimizer']['best_epoch']
        logging.info('load checkpoint {} with OA={:.4f}, epoch={}'.format(args.checkpoint, best_OA, epoch))

    # train - validation loop

    while True:
        epoch += 1
        # apportion epochs to each gpu averagely
        if epoch > int(CFG.EPOCHS / args.world_size):
            logging.info("Best epoch:{}, OA:{:.3f}".format(best_epoch, best_OA))
            if dist.get_rank() == 0:
                writer.close()
            return

        train_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        DANN.train()  # set model to training mode
        # metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='training', ascii=True)
        epoch_loss = 0.
        for train_item, test_item in zip(train_bar, test_dataloader):
            iteration += 1
            p = float(iteration + epoch * len(train_dataloader))/CFG.EPOCHS/len(train_dataloader)
            alpha = 2./(1. + np.exp(-10 * p)) - 1

            x_s, label = train_item
            x_t, _ = test_item
            domain_label_s = torch.zeros(len(label))
            domain_label_t = torch.ones(len(label))
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)
            domain_label_s, domain_label_t = domain_label_s.to(device), domain_label_t.to(device)

            optimizer.zero_grad()
            class_out, domain_out_s = DANN(x_s, alpha)
            _, domain_out_t = DANN(x_t, alpha)
            loss = train_criterion1(class_out, label, domain_out_s, domain_label_s, domain_out_t, domain_label_t)
            epoch_loss += loss.item()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            if dist.get_rank() == 0:
                writer.add_scalar('train/loss_pretrain-iteration', loss.item(), iteration)

            pred = class_out.argmax(axis=1)
            # metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
            oa, aa, kappa, per_class_acc = get_criteria(pred, label, NUM_CLASSES)

            train_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{loss.item():.3f}',
                'OA': f'{oa:.3f}',
                'AA': f'{aa:.3f}',
                'Kappa': f'{kappa:.3f}'
            })

        epoch_loss /= len(train_dataloader)
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss-epoch', epoch_loss, epoch)
            writer.add_scalar('train/OA-epoch', oa, epoch)
            writer.add_scalar('train/AA-epoch', aa, epoch)
            writer.add_scalar('train/Kappa-epoch', kappa, epoch)
        logging.info(
            'rank{} train epoch={} | loss={:.3f} '.format(dist.get_rank() + 1,epoch, epoch_loss))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} Per_class_acc={:.3f} '.format(dist.get_rank() + 1, epoch, c,
                                                                                per_class_acc[c]))

        # validate
        if args.no_validate:
            continue
        DANN.eval()  # set model to evaluation mode
        # metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label in val_bar:
                x_t, label = x_t.to(device), label.to(device)
                y_t, _ = DANN(x_t, alpha=0)   # val阶段网络不需要反传，所以alpha=0

                loss = val_criterion(y_t, label)
                val_loss += loss.item()

                pred = y_t.argmax(axis=1)
                oa, aa, kappa, per_class_acc = get_criteria(pred, label, NUM_CLASSES)

                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{loss.item():.3f}',
                    'OA': f'{oa:.3f}',
                    'AA': f'{aa:.3f}',
                    'Kappa': f'{kappa:.3f}'
                })
        val_loss /= len(val_dataloader)

        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/OA-epoch', oa, epoch)
            writer.add_scalar('val/AA-epoch', aa, epoch)
            writer.add_scalar('val/Kappa-epoch', kappa, epoch)
        if oa > best_OA:
            best_epoch = epoch

        logging.info(
            'rank{} val epoch={} | loss={:.3f} OA={:.3f} AA={:.3f} Kappa={:.3f}'.format(dist.get_rank() + 1, epoch,
                                                                                        val_loss, oa, aa, kappa))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} |  class={}- Per_class_acc={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                               per_class_acc[c]))

        # adjust learning rate if specified
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'DANN': {
                    'state_dict': DANN.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'metric': {
                    'OA': oa,
                    'AA': aa,
                    'Kappa': kappa,
                    'Per_class_acc': per_class_acc,
                },
            }
            torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
            if oa > best_OA:
                best_OA = oa
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))


def main():
    # parse command line arguments
    args = parse_args()


    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
