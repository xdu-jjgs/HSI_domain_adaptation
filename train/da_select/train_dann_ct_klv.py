import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F
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
from collections import Counter
from criterions import build_criterion
from optimizers import build_optimizer
from schedulers import build_scheduler
from models.backbone import ImageClassifier
from tllib.alignment.cdan import RandomizedMultiLinearMap

from datas import build_dataset, build_dataloader, build_iterator, DynamicDataset


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
    select_dataset = DynamicDataset()
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
    dann = build_model(NUM_CHANNELS, NUM_CLASSES)
    dann.to(device)
    mapping = RandomizedMultiLinearMap(dann.out_channels, NUM_CLASSES, output_dim=512)
    selector = ImageClassifier(dann.out_channels, 2)
    selector.to(device)
    classifier_t = ImageClassifier(dann.out_channels, NUM_CLASSES)
    classifier_t.to(device)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    domain_criterion = build_criterion(loss_names[1])
    domain_criterion.to(device)
    klv_criterion = build_criterion('kldiv')
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric_model = Metric(NUM_CLASSES)
    metric_selector = Metric(NUM_CLASSES)
    metric_ct = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_model = build_optimizer(dann)
    optimizer_selector = build_optimizer(selector)
    optimizer_ct = build_optimizer(classifier_t)
    # build scheduler
    scheduler_model = build_scheduler(optimizer_model)
    scheduler_selector = build_scheduler(optimizer_selector)
    scheduler_ct = build_scheduler(optimizer_ct)

    # mixed precision
    [dann, selector, classifier_t], [optimizer_model, optimizer_selector, optimizer_ct] = amp.initialize(
        [dann, selector, classifier_t],
        [optimizer_model,
         optimizer_selector,
         optimizer_ct],
        opt_level=args.opt_level)
    # DDP
    dann = DistributedDataParallel(dann)
    selector = DistributedDataParallel(selector)
    classifier_t = DistributedDataParallel(classifier_t)

    epoch = 0
    iteration = 0
    best_epoch_model = 0
    best_PA_model = 0.
    best_PA_ct = 0.

    # load checkpoint if specified
    if args.checkpoint is not None:
        if not os.path.isfile(args.checkpoint):
            raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        dann.load_state_dict(checkpoint['model']['state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer_model']['state_dict'])
        epoch = checkpoint['optimizer_model']['epoch']
        iteration = checkpoint['optimizer_model']['iteration']
        best_PA_model = checkpoint['metric']['PA']
        best_epoch_model = checkpoint['optimizer_model']['best_epoch']
        logging.info('load checkpoint {} with PA={:.4f}, epoch={}'.format(args.checkpoint, best_PA_model, epoch))

    # train - validation loop

    while True:
        epoch += 1
        # apportion epochs to each gpu averagely
        if epoch > int(CFG.EPOCHS / args.world_size):
            logging.info("Best model epoch:{}, PA:{:.3f}".format(best_epoch_model, best_PA_model))
            logging.info("Best ct epoch:{}, PA:{:.3f}".format(best_epoch_ct, best_PA_ct))
            if dist.get_rank() == 0:
                writer.close()
            return

        source_dataloader.sampler.set_epoch(epoch)

        if dist.get_rank() == 0:
            lr = optimizer_model.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        metric_model.reset()  # reset metric
        metric_selector.reset()
        metric_ct.reset()
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        step1_loss_epoch, cls_loss_epoch, domain_s_loss_epoch, domain_t_loss_epoch = 0., 0., 0., 0.
        selector_loss_epoch = 0.
        select_dataset.flush()
        dann.train()
        classifier_t.train()
        selector.train()
        for iteration in train_bar:
            p = float(iteration + epoch * len(source_dataloader)) / CFG.EPOCHS / len(source_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            x_s, label_s = next(source_iterator)
            x_t, label_t = next(target_iterator)
            x_s, label_s = x_s.to(device), label_s.to(device)
            x_t, label_t = x_t.to(device), label_t.to(device)

            # step1: train model
            f_s, y_s, domain_out_s = dann(x_s, alpha)
            f_t, y_t, domain_out_t = dann(x_t, alpha)

            domain_label_s = torch.zeros(len(label_s))
            domain_label_t = torch.ones(len(label_s))
            domain_label_s, domain_label_t = domain_label_s.to(device), domain_label_t.to(device)

            cls_loss = cls_criterion(y_s=y_s, label_s=label_s) * loss_weights[0]
            domain_s_loss = domain_criterion(y_s=domain_out_s, label_s=domain_label_s) * loss_weights[1]
            domain_t_loss = domain_criterion(y_s=domain_out_t, label_s=domain_label_t) * loss_weights[1]
            step1_loss = cls_loss + domain_s_loss + domain_t_loss

            cls_loss_epoch += cls_loss.item()
            domain_s_loss_epoch += domain_s_loss.item()
            domain_t_loss_epoch += domain_t_loss.item()
            step1_loss_epoch += step1_loss.item()

            if dist.get_rank() == 0:
                writer.add_scalar('train/loss_total', step1_loss.item(), iteration)
                writer.add_scalar('train/loss_cls', cls_loss.item(), iteration)
                writer.add_scalar('train/loss_domain_s', domain_s_loss.item(), iteration)
                writer.add_scalar('train/loss_domain_t', domain_t_loss.item(), iteration)

            optimizer_model.zero_grad()
            with amp.scale_loss(step1_loss, optimizer_model) as scaled_loss:
                scaled_loss.backward()
            optimizer_model.step()

            pred = y_s.argmax(axis=1)
            metric_model.add(pred.data.cpu().numpy(), label_s.data.cpu().numpy())

            train_bar.set_postfix({
                'iteration': iteration,
                'epoch': epoch,
                'loss': f'{step1_loss.item():.3f}',
                'mP': f'{metric_model.mPA():.3f}',
                'PA': f'{metric_model.PA():.3f}',
                'KC': f'{metric_model.KC():.3f}'
            })

            # step2: train selector
            dann.eval()
            classifier_t.eval()
            with torch.no_grad():
                f_s, y_s, domain_out_s = dann(x_s, alpha)
                f_t, y_t, domain_out_t = dann(x_t, alpha)
                f_t = torch.squeeze(f_t, dim=-1)
                f_t = torch.squeeze(f_t, dim=-1)
                y_t_ = F.softmax(y_t, dim=1).detach()
                joint = mapping(f_t, y_t_)
            select_status = selector(joint)[-1]

            # 先试试接近
            domain_out_t_ = domain_out_t.argmin(axis=1)
            confu_loss = cls_criterion(y_s=select_status, label_s=domain_out_t_)
            y_s_c_t = classifier_t(f_s)[-1]
            cls_loss_ct_xs = cls_criterion(y_s=y_s_c_t, label_s=label_s)
            step2_loss = confu_loss + cls_loss_ct_xs
            # step2_loss = confu_loss
            selector_loss_epoch += step2_loss.item()

            optimizer_model.zero_grad()
            optimizer_selector.zero_grad()
            optimizer_ct.zero_grad()
            with amp.scale_loss(step2_loss, optimizer_selector) as scaled_loss:
                scaled_loss.backward()
            optimizer_selector.step()

            select_mask = select_status.argmax(axis=1)
            x_t_select = x_t[select_mask == 1]
            y_t_select = y_t_[select_mask == 1]
            label_t_select = label_t[select_mask == 1]
            for d, l in zip(x_t_select, y_t_select):
                select_dataset.append(d, l)
            metric_selector.add(y_t_select.argmax(axis=1).data.cpu().numpy(), label_t_select.data.cpu().numpy())

        selector.eval()
        classifier_t.train()
        logging.info("Num of selected data: ", len(select_dataset))
        pseudo_labels = list(map(lambda x: x.argmax(axis=0).item(), select_dataset.gt))
        logging.info("Count: ", Counter(pseudo_labels))
        selected_sampler = DistributedSampler(select_dataset, shuffle=False)
        selected_dataloder = build_dataloader(select_dataset, sampler=selected_sampler)
        selected_bar = tqdm(selected_dataloder, desc='training_ct', ascii=True)
        # TODO: try to use soft label?
        # TODO: try train dann?
        for x_t, pseudo_labels_t in selected_bar:
            with torch.no_grad():
                f_t, _, _ = dann(x_t, alpha=0)
            y_t_ct = classifier_t(f_t)[-1]
            step3_loss = klv_criterion(y_t_ct, pseudo_labels_t)

            optimizer_ct.zero_grad()
            with amp.scale_loss(step3_loss, optimizer_ct) as scaled_loss:
                scaled_loss.backward()
            optimizer_ct.step()

            pred_ct = y_t_ct.argmax(axis=1)
            metric_ct.add(pred_ct.data.cpu().numpy(), pseudo_labels_t.argmax(axis=1).data.cpu().numpy())

        step1_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_s_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_t_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        PA, mPA, Ps, Rs, F1S, KC = metric_model.PA(), metric_model.mPA(), metric_model.Ps(), metric_model.Rs(), metric_model.F1s(), metric_model.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_total-epoch', step1_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls-epoch', cls_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_s-epoch', domain_s_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_t-epoch', domain_t_loss_epoch, epoch)
            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)
        logging.info(
            'rank{} train epoch={} | loss_total={:.3f} loss_cls={:.3f} loss_domain_s={:.3f} loss_domain_t={:.3f}'.format(
                dist.get_rank() + 1, epoch, step1_loss_epoch, cls_loss_epoch, domain_s_loss_epoch, domain_t_loss_epoch))
        logging.info(
            'rank{} train epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                      Ps[c], Rs[c], F1S[c]))
        PA, mPA, Ps, Rs, F1S, KC = metric_selector.PA(), metric_selector.mPA(), metric_selector.Ps(), metric_selector.Rs(), metric_selector.F1s(), metric_selector.KC()
        logging.info(
            'rank{} train selector epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train selector epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                          Ps[c], Rs[c], F1S[c]))
        PA, mPA, Ps, Rs, F1S, KC = metric_ct.PA(), metric_ct.mPA(), metric_ct.Ps(), metric_ct.Rs(), metric_ct.F1s(), metric_ct.KC()
        logging.info(
            'rank{} train CT epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train CT epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                          Ps[c], Rs[c], F1S[c]))

        # validate
        if args.no_validate:
            continue
        dann.eval()  # set model to evaluation mode
        selector.eval()
        classifier_t.eval()
        metric_model.reset()  # reset metric
        metric_ct.reset()
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label_t in val_bar:
                x_t, label_t = x_t.to(device), label_t.to(device)
                f_t, y_t, _ = dann(x_t, alpha=0)  # val阶段网络不需要反传，所以alpha=0

                y_t_ct = classifier_t(f_t)[-1]
                pred_ct = y_t_ct.argmax(axis=-1)
                metric_ct.add(pred_ct.data.cpu().numpy(), label_t.data.cpu().numpy())

                pred = y_t.argmax(axis=1)
                metric_model.add(pred.data.cpu().numpy(), label_t.data.cpu().numpy())

                cls_loss = val_criterion(y_s=y_t, label_s=label_t)
                val_loss += cls_loss.item()
                val_bar.set_postfix({
                    'epoch': epoch,
                    'loss': f'{cls_loss.item():.3f}',
                    'mP': f'{metric_model.mPA():.3f}',
                    'PA': f'{metric_model.PA():.3f}',
                    'KC': f'{metric_model.KC():.3f}'
                })

        val_loss /= len(val_dataloader)

        PA, mPA, Ps, Rs, F1S, KC = metric_model.PA(), metric_model.mPA(), metric_model.Ps(), metric_model.Rs(), metric_model.F1s(), metric_model.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/PA-epoch', PA, epoch)
            writer.add_scalar('val/mPA-epoch', mPA, epoch)
            writer.add_scalar('val/KC-epoch', KC, epoch)
        if PA > best_PA_model:
            best_epoch_model = epoch

        logging.info('rank{} val epoch={} | loss={:.3f}'.format(dist.get_rank() + 1, epoch, val_loss))
        logging.info(
            'rank{} val epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} | class={}- P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                     Ps[c], Rs[c], F1S[c]))

        PA_ct, mPA, Ps, Rs, F1S, KC = metric_ct.PA(), metric_ct.mPA(), metric_ct.Ps(), metric_ct.Rs(), metric_ct.F1s(), metric_ct.KC()
        if PA_ct > best_PA_ct:
            best_epoch_ct = epoch
        logging.info(
            'rank{} val epoch={} | CT: PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} | CT: class={}- P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                         Ps[c], Rs[c], F1S[c]))

        # adjust learning rate if specified
        if scheduler_model is not None:
            try:
                scheduler_model.step()
                scheduler_selector.step()
                scheduler_ct.step()
            except TypeError:
                scheduler_model.step(val_loss)
                scheduler_selector.step(val_loss)
                scheduler_ct.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'model': {
                    'state_dict': dann.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer_model.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch_model,
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
            if PA > best_PA_model:
                best_PA_model = PA
                torch.save(checkpoint, os.path.join(args.path, 'best.pth'))
            if PA_ct > best_PA_ct:
                best_PA_ct = PA_ct
                torch.save(checkpoint, os.path.join(args.path, 'best_ct.pth'))


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
