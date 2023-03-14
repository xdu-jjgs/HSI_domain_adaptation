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
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    # print(model)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    wcec_criterion = build_criterion(loss_names[1])
    cbst_criterion = build_criterion(loss_names[2])
    cls_criterion.to(device)
    wcec_criterion.to(device)
    cbst_criterion.to(device)
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric_cls = Metric(NUM_CLASSES)
    metric_adv_s = Metric(NUM_CLASSES)
    metric_adv_t = Metric(NUM_CLASSES)
    metric_pse = Metric(NUM_CLASSES)
    # build optimizer
    optimizer = build_optimizer(model)
    # build scheduler
    scheduler = build_scheduler(optimizer)

    # mixed precision
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # DDP
    model = DistributedDataParallel(model)

    epoch = 0
    iteration = 0
    best_epoch = 0
    best_PA = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model']['state_dict'])
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
        model.train()  # set model to training mode
        metric_cls.reset()  # reset metric
        metric_adv_s.reset()
        metric_adv_t.reset()
        metric_pse.reset()
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        total_loss_epoch, cls_loss_epoch, worst_loss_epoch, cbst_loss_epoch = 0., 0., 0., 0.,
        for iteration in train_bar:
            x_s, label_s = next(source_iterator)
            x_t, label_t = next(target_iterator)
            x_s, label_s = x_s.to(device), label_s.to(device)
            x_t = x_t.to(device)

            # step1: train classifier with S data
            f_s, y_s, _, y_s_adv = model(x_s)
            f_t, y_t, y_pse, y_t_adv = model(x_t)

            cls_loss = cls_criterion(y_s=y_s, label_s=label_s,) * loss_weights[0]
            worst_loss = wcec_criterion(y_s, y_s_adv, y_t, y_t_adv) * loss_weights[1]
            cls_loss_epoch += cls_loss.item()
            worst_loss_epoch += worst_loss.item()

            # step2: train classifier_pse with T data
            cbst_loss, mask, labels_pse = cbst_criterion(y_pse, y_t)
            cbst_loss *= loss_weights[2]
            cbst_loss_epoch += cbst_loss.item()

            total_loss = cls_loss + worst_loss + cbst_loss
            total_loss_epoch += total_loss.item()

            optimizer.zero_grad()
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # update metric for cls, cls_adv and cls_pse
            pred_cls = y_s.argmax(axis=1)
            pred_adv_s = y_s_adv.argmax(axis=1)
            pred_adv_t = y_t_adv.argmax(axis=1)
            metric_cls.add(pred_cls.data.cpu().numpy(), label_s.data.cpu().numpy())
            metric_adv_s.add(pred_adv_s.data.cpu().numpy(), label_s.data.cpu().numpy())
            metric_adv_t.add(pred_adv_t.data.cpu().numpy(), label_s.data.cpu().numpy())
            pred_pse = y_pse.argmax(axis=1)
            metric_pse.add(pred_pse.data.cpu().numpy(), label_t.data.cpu().numpy())

            if dist.get_rank() == 0:
                writer.add_scalar('train/loss_total', total_loss.item(), iteration)
                writer.add_scalar('train/loss_cls', cls_loss.item(), iteration)
                writer.add_scalar('train/loss_wos', worst_loss.item(), iteration)
                writer.add_scalar('train/loss_cbst', cbst_loss.item(), iteration)

            train_bar.set_postfix({
                'epoch': epoch,
                'total-loss': f'{total_loss.item():.3f}',
                'mP': f'{metric_cls.mPA():.3f}',
                'PA': f'{metric_cls.PA():.3f}',
                'KC': f'{metric_cls.KC():.3f}',
            })

        total_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        worst_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cbst_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE

        PA_cls, mPA_cls, Ps_cls, Rs_cls, F1S_cls, KC_cls = \
            metric_cls.PA(), metric_cls.mPA(), metric_cls.Ps(), metric_cls.Rs(), metric_cls.F1s(), metric_cls.KC()
        PA_adv_s, mPA_adv_s, Ps_adv_s, Rs_adv_s, F1S_adv_s, KC_adv_s = \
            metric_adv_s.PA(), metric_adv_s.mPA(), metric_adv_s.Ps(), \
            metric_adv_s.Rs(), metric_adv_s.F1s(), metric_adv_s.KC()
        PA_adv_t, mPA_adv_t, Ps_adv_t, Rs_adv_t, F1S_adv_t, KC_adv_t = \
            metric_adv_t.PA(), metric_adv_t.mPA(), metric_adv_t.Ps(), \
            metric_adv_t.Rs(), metric_adv_t.F1s(), metric_adv_t.KC()
        PA_pse, mPA_pse, Ps_pse, Rs_pse, F1S_pse, KC_pse = \
            metric_pse.PA(), metric_pse.mPA(), metric_pse.Ps(), metric_pse.Rs(), metric_pse.F1s(), metric_pse.KC()

        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_total-epoch', total_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls-epoch', cls_loss_epoch, epoch)
            writer.add_scalar('train/loss_wos-epoch', worst_loss_epoch, epoch)

            writer.add_scalar('train/PA_cls-epoch', PA_cls, epoch)
            writer.add_scalar('train/mPA_cls-epoch', mPA_cls, epoch)
            writer.add_scalar('train/KC_cls-epoch', KC_cls, epoch)

            writer.add_scalar('train/PA_adv_s-epoch', PA_adv_s, epoch)
            writer.add_scalar('train/mPA_adv_s-epoch', mPA_adv_s, epoch)
            writer.add_scalar('train/KC_adv_s-epoch', KC_adv_s, epoch)

            writer.add_scalar('train/PA_adv_t-epoch', PA_adv_t, epoch)
            writer.add_scalar('train/mPA_adv_t-epoch', mPA_adv_t, epoch)
            writer.add_scalar('train/KC_adv_t-epoch', KC_adv_t, epoch)

            writer.add_scalar('train/PA_pse-epoch', PA_pse, epoch)
            writer.add_scalar('train/mPA_pse-epoch', mPA_pse, epoch)
            writer.add_scalar('train/KC_pse-epoch', KC_pse, epoch)
        logging.info(
            'rank{} train epoch={} | loss_total={:.3f} loss_cls={:.3f} loss_wos={:.3f} loss_cbst={:.3f}'
            .format(dist.get_rank() + 1, epoch, total_loss_epoch, cls_loss_epoch, worst_loss_epoch, cbst_loss_epoch))
        logging.info(
            'rank{} train epoch={} | cls PA={:.3f} mPA={:.3f} KC={:.3f} | adv-s PA={:.3f} mPA={:.3f} KC={:.3f} |'
            ' adv_t PA={:.3f} mPA={:.3f} KC={:.3f}| pse PA={:.3f} mPA={:.3f} KC={:.3f}'
            .format(dist.get_rank() + 1, epoch, PA_cls, mPA_cls, KC_cls, PA_adv_s, mPA_adv_s, KC_adv_s, PA_adv_t,
                    mPA_adv_t, KC_adv_t, PA_pse, mPA_pse, KC_pse))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} | cls P={:.3f} R={:.3f} F1={:.3f}|adv_s P={:.3f} R={:.3f} F1={:.3f}|'
                ' adv_t P={:.3f} R={:.3f} F1={:.3f}| pse P={:.3f} R={:.3f} F1={:.3f}'
                .format(dist.get_rank() + 1, epoch, c + 1, Ps_cls[c], Rs_cls[c], F1S_cls[c], Ps_adv_s[c], Rs_adv_s[c],
                        F1S_adv_s[c], Ps_adv_t[c], Rs_adv_t[c], F1S_adv_t[c], Ps_pse[c], Rs_pse[c], F1S_pse[c]))

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        metric_cls.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label_s in val_bar:
                x_t, label_s = x_t.to(device), label_s.to(device)
                _, y_t, _, _ = model(x_t)

                cls_loss = val_criterion(y_s=y_t, label_s=label_s)
                val_loss += cls_loss.item()

                pred = y_t.argmax(axis=1)
                metric_cls.add(pred.data.cpu().numpy(), label_s.data.cpu().numpy())

                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{cls_loss.item():.3f}',
                    'mP': f'{metric_cls.mPA():.3f}',
                    'PA': f'{metric_cls.PA():.3f}',
                    'KC': f'{metric_cls.KC():.3f}'
                })
        val_loss /= len(val_dataloader)

        PA, mPA, Ps, Rs, F1S, KC = metric_cls.PA(), metric_cls.mPA(), metric_cls.Ps(), metric_cls.Rs(), metric_cls.F1s(), metric_cls.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/PA-epoch', PA, epoch)
            writer.add_scalar('val/mPA-epoch', mPA, epoch)
            writer.add_scalar('val/KC-epoch', KC, epoch)
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
                'model': {
                    'state_dict': model.state_dict(),
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


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
