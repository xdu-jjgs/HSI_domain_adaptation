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
from torchsampler import ImbalancedDatasetSampler
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
    # test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_sampler = ImbalancedDatasetSampler(test_dataset, labels=test_dataset.get_labels())
    # build data loader
    train_dataloader = build_dataloader(train_dataset, sampler=train_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)
    test_dataloader = build_dataloader(test_dataset, sampler=test_sampler)

    # build model TODO: notice here
    G, D, C = build_model(NUM_CHANNELS, NUM_CLASSES)
    G.to(device)
    D.to(device)
    C.to(device)
    # print(model)
    # build criterion TODO: notice here
    train_criterion1 = build_criterion(CFG.CRITERION.ITEMS1, CFG.CRITERION.WEIGHTS1)
    train_criterion1.to(device)
    train_criterion2 = build_criterion(CFG.CRITERION.ITEMS2, CFG.CRITERION.WEIGHTS2)
    train_criterion2.to(device)
    train_criterion3 = build_criterion(CFG.CRITERION.ITEMS3, CFG.CRITERION.WEIGHTS3)
    train_criterion3.to(device)
    val_criterion = build_criterion(CFG.CRITERION.ITEMS4, CFG.CRITERION.WEIGHTS4)
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_g = build_optimizer(G, CFG.OPTIMIZER.NAME1, CFG.OPTIMIZER.LR1)
    optimizer_d = build_optimizer(D, CFG.OPTIMIZER.NAME2, CFG.OPTIMIZER.LR2)
    optimizer_c = build_optimizer(C, CFG.OPTIMIZER.NAME3, CFG.OPTIMIZER.LR3)
    # build scheduler
    scheduler1 = build_scheduler(optimizer_g)
    scheduler2 = build_scheduler(optimizer_d)
    scheduler3 = build_scheduler(optimizer_c)

    # mixed precision
    [G, D, C], [optimizer_g, optimizer_d, optimizer_c] = amp.initialize([G, D, C],
                                                                        [optimizer_g, optimizer_d, optimizer_c],
                                                                        opt_level=args.opt_level)
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
        optimizer_g.load_state_dict(checkpoint['optimizer_g']['state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d']['state_dict'])
        optimizer_c.load_state_dict(checkpoint['optimizer_c']['state_dict'])
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
            lr_g = optimizer_g.param_groups[0]['lr']
            lr_d = optimizer_d.param_groups[0]['lr']
            lr_c = optimizer_c.param_groups[0]['lr']
            writer.add_scalar('lrg-epoch', lr_g, epoch)
            writer.add_scalar('lrd-epoch', lr_d, epoch)
            writer.add_scalar('lrc-epoch', lr_c, epoch)

        # train
        category2cnt = torch.zeros(NUM_CLASSES).to(device)
        G.train()  # set model to training mode
        D.train()  # set model to training mode
        C.train()  # set model to training mode
        # metric.reset()  # reset metric
        train_bar = tqdm(train_dataloader, desc='training', ascii=True)
        epoch_g_loss = 0.
        epoch_d_loss = 0.
        epoch_c_loss = 0.
        for train_item, test_item in zip(train_bar, test_dataloader):
            iteration += 1
            x_s, label = train_item
            x_t, _ = test_item
            z = torch.rand([len(label), NUM_CHANNELS], dtype=torch.float)
            z_info = torch.tensor(one_hot(label, NUM_CLASSES), dtype=torch.float32)
            # label_fake = NUM_CLASSES * torch.ones(len(label), dtype=torch.int32)
            onehot_label = torch.eye(NUM_CLASSES)[label.long(), :]
            domain_label_fake = torch.zeros(len(label))
            domain_label_t = torch.ones(len(label))
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)
            z, z_info = z.to(device), z_info.to(device)
            onehot_label = onehot_label.to(device)
            # label_fake = label_fake.to(device)
            domain_label_fake, domain_label_t = domain_label_fake.to(device), domain_label_t.to(device)

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            optimizer_c.zero_grad()

            # train C
            x_fake = G(z, z_info, x_s)
            c_fake = C(x_fake)
            c_s = C(x_s)
            # Default: ce loss, when margin loss change label to onehot_label
            loss_c = train_criterion1(c_s, label, c_fake, label)
            # loss_c = train_criterion1(c_s, onehot_label, c_fake, onehot_label)
            epoch_c_loss += loss_c.item()
            with amp.scale_loss(loss_c, optimizer_c) as scaled_loss:
                scaled_loss.backward()
            optimizer_c.step()

            optimizer_d.zero_grad()
            # train D maximize log(D(x)) + log(1 - D(G(z)))
            x_fake = G(z, z_info, x_s)
            _, d_fake = D(x_fake)
            _, d_t = D(x_t)
            loss_d = train_criterion2(d_t, domain_label_t, d_fake, domain_label_fake)
            epoch_d_loss += loss_d.item()
            with amp.scale_loss(loss_d, optimizer_d) as scaled_loss:
                scaled_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad()
            # train g
            x_fake = G(z, z_info, x_s)
            c_fake = C(x_fake)
            c_t = C(x_t)
            for category in c_t.argmax(axis=1):
                category2cnt[category] += 1
            _, d_fake = D(x_fake)
            # Default: ce loss, when margin loss change label to onehot_label
            loss_g = train_criterion3(c_fake, label, c_fake, c_t, d_fake, domain_label_t)
            # loss_g = train_criterion3(c_fake, onehot_label, c_fake, c_t, d_fake, domain_label_t)
            epoch_g_loss += loss_g.item()
            with amp.scale_loss(loss_g, optimizer_g) as scaled_loss:
                scaled_loss.backward()
            optimizer_g.step()

            if dist.get_rank() == 0:
                writer.add_scalar('train/loss_g-iteration', loss_g.item(), iteration)
                writer.add_scalar('train/loss_c-iteration', loss_c.item(), iteration)
                writer.add_scalar('train/loss_d-iteration', loss_d.item(), iteration)

            pred = c_s.argmax(axis=1)
            # metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
            oa, aa, kappa, per_class_acc = get_criteria(pred, label, NUM_CLASSES)

            train_bar.set_postfix({
                'epoch': epoch,
                'loss_g': f'{loss_g.item():.3f}',
                'loss_c': f'{loss_c.item():.3f}',
                'loss_d': f'{loss_d.item():.3f}',
                'OA': f'{oa:.3f}',
                'AA': f'{aa:.3f}',
                'Kappa': f'{kappa:.3f}'
            })

        epoch_g_loss /= len(train_dataloader)
        epoch_c_loss /= len(train_dataloader)
        epoch_d_loss /= len(train_dataloader)
        if dist.get_rank() == 0:
            writer.add_scalar('train/g_loss-epoch', epoch_g_loss, epoch)
            writer.add_scalar('train/c_loss-epoch', epoch_c_loss, epoch)
            writer.add_scalar('train/d_loss-epoch', epoch_d_loss, epoch)
            writer.add_scalar('train/OA-epoch', oa, epoch)
            writer.add_scalar('train/AA-epoch', aa, epoch)
            writer.add_scalar('train/Kappa-epoch', kappa, epoch)
        logging.info(
            'rank{} train epoch={} | g_loss={:.3f} c_loss={:.3f} d_loss={:.3f}'.format(dist.get_rank() + 1,
                                                                                       epoch, epoch_g_loss,
                                                                                       epoch_c_loss, epoch_d_loss))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} Per_class_acc={:.3f} '.format(dist.get_rank() + 1, epoch, c,
                                                                                per_class_acc[c]))

        logging.info(category2cnt)

        # validate
        if args.no_validate:
            continue
        G.eval()  # set model to evaluation mode
        D.eval()  # set model to evaluation mode
        C.eval()  # set model to evaluation mode
        # TODO attention:如果 retain graph = true 此处不能用eval() 否则计算图会被free掉 导致模型失效
        # metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label in val_bar:
                x_t, label = x_t.to(device), label.to(device)
                y_t = C(x_t)

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
        if scheduler1 is not None:
            try:
                scheduler1.step()
                scheduler2.step()
                scheduler3.step()
            except TypeError:
                scheduler1.step(val_loss)
                scheduler2.step(val_loss)
                scheduler3.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'G': {
                    'state_dict': G.state_dict(),
                },
                'D': {
                    'state_dict': D.state_dict(),
                },
                'C': {
                    'state_dict': C.state_dict(),
                },
                'optimizer_g': {
                    'state_dict': optimizer_g.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'optimizer_d': {
                    'state_dict': optimizer_d.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'optimizer_c': {
                    'state_dict': optimizer_c.state_dict(),
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
