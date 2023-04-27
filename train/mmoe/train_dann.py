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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from models import build_model
from criterions import build_criterion
from optimizers import build_optimizer
from schedulers import build_scheduler
from datas import build_dataset, build_dataloader, build_iterator


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
    args.path = os.path.join(args.path, str(args.seed))
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
    source_dataset = build_dataset('train')
    target_dataset = build_dataset('test')
    val_dataset = target_dataset
    assert source_dataset.num_classes == val_dataset.num_classes
    logging.info(
        "Number of train {}, val {}, test {}".format(len(source_dataset), len(val_dataset), len(target_dataset)))
    NUM_CHANNELS = source_dataset.num_channels
    NUM_CLASSES = source_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    # build data sampler
    source_sampler = DistributedSampler(source_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    target_sampler = DistributedSampler(target_dataset, shuffle=True)
    # build data loader
    source_dataloader = build_dataloader(source_dataset, sampler=source_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)
    target_dataloader = build_dataloader(target_dataset, sampler=target_sampler)
    # build data iteration
    source_iterator = build_iterator(source_dataloader)
    target_iterator = build_iterator(target_dataloader)
    # build mmoe
    # -2 for adding reverse layer
    mmoe = build_model(NUM_CHANNELS, [NUM_CLASSES, -2])
    mmoe.to(device)
    # print(mmoe)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    domain_criterion = build_criterion(loss_names[1])
    domain_criterion.to(device)
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(mmoe)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    # mixed precision
    mmoe, optimizer = amp.initialize(mmoe, optimizer, opt_level=args.opt_level)
    # DDP
    mmoe = DistributedDataParallel(mmoe, broadcast_buffers=False, find_unused_parameters=True)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_PA = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        mmoe.load_state_dict(checkpoint['mmoe']['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer']['state_dict'])
        epoch = checkpoint['optimizer']['epoch']
        iteration = checkpoint['optimizer']['iteration']
        best_PA = checkpoint['metric']['PA']
        best_epoch = checkpoint['optimizer']['best_epoch']
        logging.info('load checkpoint {} with PA={:.4f}, epoch={}'.format(args.checkpoint, best_PA, epoch))

    # train - validation loop

    while True:
        epoch += 1
        # apportion epochs to each gpu averagely
        if epoch > int(CFG.EPOCHS / args.world_size):
            logging.info("Best epoch:{}, PA:{:.3f}".format(best_epoch, best_PA))
            if dist.get_rank() == 0:
                writer.close()
            return

        source_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        # 一个K分类器，一个mmd对齐
        mmoe.train()  # set mmoe to training mode
        metric.reset()  # reset metric
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        total_loss_epoch, cls_loss_epoch, domain_s_loss_epoch, domain_t_loss_epoch = 0., 0., 0., 0.
        cls_weight_epoch, domain_weight_epoch = np.zeros((len(mmoe.module.experts))), np.zeros(
            (len(mmoe.module.experts)))
        for iteration in train_bar:
            x_s, label = next(source_iterator)
            x_t, _ = next(target_iterator)
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)
            domain_s_label = torch.zeros(len(label))
            domain_t_label = torch.ones(len(label))
            domain_s_label, domain_t_label = domain_s_label.to(device), domain_t_label.to(device)

            out_s_cls, out_s_domain, task_weights = mmoe(x_s)
            _, y_s_cls = out_s_cls
            _, domain_s_out = out_s_domain
            cls_weight_epoch += task_weights[0].sum(dim=0).squeeze(0).detach().cpu().numpy()
            domain_weight_epoch += task_weights[1].sum(dim=0).squeeze(0).detach().cpu().numpy()

            _, out_t_dis, task_weights = mmoe(x_t)
            _, domain_t_out = out_t_dis
            domain_weight_epoch += task_weights[1].sum(dim=0).squeeze(0).detach().cpu().numpy()

            cls_loss = cls_criterion(label_s=label, y_s=y_s_cls) * loss_weights[0]
            domain_s_loss = domain_criterion(y_s=domain_s_out, label_s=domain_s_label) * loss_weights[1]
            domain_t_loss = domain_criterion(y_s=domain_t_out, label_s=domain_t_label) * loss_weights[1]
            total_loss = cls_loss + domain_s_loss + domain_t_loss

            cls_loss_epoch += cls_loss.item()
            domain_s_loss_epoch += domain_s_loss.item()
            domain_t_loss_epoch += domain_t_loss.item()
            total_loss_epoch += total_loss.item()

            optimizer.zero_grad()
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            pred = y_s_cls.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

            train_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{total_loss.item():.3f}',
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}',
            })

        total_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_s_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_t_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_weight_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_weight_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE * 2
        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_total-epoch', total_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls-epoch', cls_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_s-epoch', domain_s_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_t-epoch', domain_t_loss_epoch, epoch)

            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)

            for ind, ele in enumerate(zip(cls_weight_epoch, domain_weight_epoch)):
                writer.add_scalar('train/cls_weight_expert{}'.format(ind+1), ele[0], epoch)
                writer.add_scalar('train/domain_weight_expert{}'.format(ind+1), ele[1], epoch)
        logging.info(
            'rank{} train epoch={} | loss_total={:.3f} loss_cls={:.3f} loss_domain_s={:.3f} loss_domain_t={:.3f}'.format(
                dist.get_rank() + 1, epoch, total_loss_epoch, cls_loss_epoch, domain_s_loss_epoch, domain_t_loss_epoch))
        logging.info(
            'rank{} train epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                      Ps[c], Rs[c], F1S[c]))

        # validate
        if args.no_validate:
            continue
        mmoe.eval()  # set mmoe to evaluation mode
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        cls_weight_epoch, domain_weight_epoch = np.zeros((len(mmoe.module.experts))), np.zeros(
            (len(mmoe.module.experts)))
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label in val_bar:
                x_t, label = x_t.to(device), label.to(device)
                out_cls, _, task_weights = mmoe(x_t)
                _, y_t = out_cls

                cls_weight_epoch += task_weights[0].sum(dim=0).squeeze(0).detach().cpu().numpy()
                domain_weight_epoch += task_weights[1].sum(dim=0).squeeze(0).detach().cpu().numpy()

                cls_loss = val_criterion(y_s=y_t, label_s=label)
                val_loss += cls_loss.item()

                pred = y_t.argmax(axis=1)
                metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{cls_loss.item():.3f}',
                    'mP': f'{metric.mPA():.3f}',
                    'PA': f'{metric.PA():.3f}',
                    'KC': f'{metric.KC():.3f}'
                })
        val_loss /= len(val_dataloader)
        cls_weight_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_weight_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        
        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)

            writer.add_scalar('val/PA-epoch', PA, epoch)
            writer.add_scalar('val/mPA-epoch', mPA, epoch)
            writer.add_scalar('val/KC-epoch', KC, epoch)

            for ind, ele in enumerate(zip(cls_weight_epoch, domain_weight_epoch)):
                writer.add_scalar('val/cls_weight_expert{}'.format(ind+1), ele[0], epoch)
                writer.add_scalar('val/domain_weight_expert{}'.format(ind+1), ele[1], epoch)
        if PA > best_PA:
            best_epoch = epoch

        logging.info('rank{} val epoch={} | loss={:.3f}'.format(dist.get_rank() + 1, epoch, val_loss))
        logging.info(
            'rank{} val epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} | class={}- P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                     Ps[c], Rs[c], F1S[c]))

        # adjust learning rate if specified
        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'mmoe': {
                    'state_dict': mmoe.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'metric': {
                    'PA': PA,
                    'mPA': mPA,
                    'Ps': Ps,
                    'Rs': Rs,
                    'F1S': F1S,
                    'KC': KC
                },
            }
            torch.save(checkpoint, os.path.join(args.path, 'last.pth'))
            if PA > best_PA:
                best_PA = PA
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))
            writer.add_scalar('best-PA', best_PA, epoch)

def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()