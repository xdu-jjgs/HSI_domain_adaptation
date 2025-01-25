import os
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

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
from models.utils.grad_filter import cal_grad_scores
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
                            init_method=f'tcp://{args.master_ip}:{args.master_port}?use_libuv=False',
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
    model, classifier, domain_classifier = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    classifier.to(device)
    domain_classifier.to(device)
    # print(model)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    similarity_criterion = build_criterion(loss_names[1])
    similarity_criterion.to(device)
    difference_criterion = build_criterion(loss_names[2])
    difference_criterion.to(device)
    domain_criterion = build_criterion(loss_names[3])
    domain_criterion.to(device)
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_model = build_optimizer(model)
    optimizer_classifier = build_optimizer(classifier)
    optimizer_domain_classifier = build_optimizer(domain_classifier)
    # build scheduler
    scheduler_model = build_scheduler(optimizer_model)
    scheduler_classifier = build_scheduler(optimizer_classifier)
    scheduler_domain_classifier = build_scheduler(optimizer_domain_classifier)
    # DDP
    model = DistributedDataParallel(model, broadcast_buffers=False)
    classifier = DistributedDataParallel(classifier)
    domain_classifier = DistributedDataParallel(domain_classifier)

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
        classifier.load_state_dict(checkpoint['classifier']['state_dict'])
        domain_classifier.load_state_dict(checkpoint['domain_classifier']['state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer']['state_dict'])
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
            lr = optimizer_model.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train

        metric.reset()  # reset metric
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        step1_loss_epoch, step2_loss_epoch, cls_s_loss_epoch, cls_t_loss_epoch, domain_s_loss_epoch, \
            domain_t_loss_epoch, difference_s_loss_epoch, difference_t_loss_epoch = \
            0., 0., 0., 0., 0., 0., 0., 0.
        for iteration in train_bar:
            x_s, label = next(source_iterator)
            x_t, _ = next(target_iterator)
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)
            domain_label_s = torch.zeros(len(label))
            domain_label_t = torch.ones(len(label))
            domain_label_s, domain_label_t = domain_label_s.to(device), domain_label_t.to(device)

            # step1: train model and cal score
            model.train()  # set model to training mode
            classifier.train()
            domain_classifier.eval()
            optimizer_model.zero_grad()
            optimizer_domain_classifier.zero_grad()
            shared_f_s, private_f_s, = model(x_s, 1)
            shared_f_t, private_f_t, = model(x_t, 2)
            domain_out_s = domain_classifier(shared_f_s)[-1]
            domain_out_t = domain_classifier(shared_f_t)[-1]
            scores_s = cal_grad_scores(shared_f_s, domain_out_s, 1)
            scores_t = cal_grad_scores(shared_f_t, domain_out_t, 2)

            filter_num = int(model.module.filter_ratio * scores_s.size()[1])
            mask_s_ds = torch.ones_like(shared_f_s).to(device)
            row = torch.arange(mask_s_ds.size()[0]).unsqueeze(1).to(device)
            _, index_s_ds_channels = scores_s.topk(filter_num)
            mask_s_ds[row, index_s_ds_channels] = 0.
            masked_ds_shared_s_features = shared_f_s * mask_s_ds
            y_s = classifier(masked_ds_shared_s_features)[-1]
            mask_t_ds = torch.ones_like(shared_f_t).to(device)
            _, index_t_ds_channels = scores_t.topk(filter_num)
            mask_t_ds[row, index_t_ds_channels] = 0.
            masked_ds_shared_t_features = shared_f_t * mask_t_ds
            y_t = classifier(masked_ds_shared_t_features)[-1]

            cls_s_loss = cls_criterion(y_s=y_s, label_s=label) * loss_weights[0]
            cls_t_loss, mask, pseudo_labels = similarity_criterion(y_t, y_t)
            cls_t_loss *= loss_weights[1]
            difference_s_loss = difference_criterion(shared_f_s, private_f_s) * loss_weights[2]
            difference_t_loss = difference_criterion(shared_f_t, private_f_t) * loss_weights[2]
            step1_loss = cls_s_loss + cls_t_loss + difference_s_loss + difference_t_loss

            step1_loss.backward()
            optimizer_model.step()
            cls_s_loss_epoch += cls_s_loss.item()
            cls_t_loss_epoch += cls_t_loss.item()
            difference_s_loss_epoch += difference_s_loss.item()
            difference_t_loss_epoch += difference_t_loss.item()
            step1_loss_epoch += step1_loss.item()

            # step2: train domain classifier
            model.eval()
            classifier.eval()
            domain_classifier.train()
            optimizer_domain_classifier.zero_grad()
            domain_out_s = domain_classifier(shared_f_s.detach())[-1]
            domain_out_t = domain_classifier(shared_f_t.detach())[-1]
            domain_s_loss = domain_criterion(y_s=domain_out_s, label_s=domain_label_s) * loss_weights[3]
            domain_t_loss = domain_criterion(y_s=domain_out_t, label_s=domain_label_t) * loss_weights[3]
            step2_loss = domain_s_loss + domain_t_loss

            step2_loss.backward()
            optimizer_domain_classifier.step()
            domain_s_loss_epoch += domain_s_loss.item()
            domain_t_loss_epoch += domain_t_loss.item()

            pred = y_s.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
            train_bar.set_postfix({
                'iteration': iteration,
                'epoch': epoch,
                'loss': f'{step1_loss.item():.3f}',
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}'
            })

        step1_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        step2_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_s_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_t_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_s_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        domain_t_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        difference_s_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        difference_t_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_s1-epoch', step1_loss_epoch, epoch)
            writer.add_scalar('train/loss_s2-epoch', step2_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls_s-epoch', cls_s_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls_t-epoch', cls_t_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_s-epoch', domain_s_loss_epoch, epoch)
            writer.add_scalar('train/loss_domain_t-epoch', domain_t_loss_epoch, epoch)
            writer.add_scalar('train/loss_difference_s-epoch', difference_s_loss_epoch, epoch)
            writer.add_scalar('train/loss_difference_t-epoch', difference_t_loss_epoch, epoch)

            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)
        logging.info(
            'rank{} train epoch={} | loss_s1={:.3f} loss_s2={:.3f} loss_cls_s={:.3f} loss_cls_t={:.3f} '
            'loss_domain_s={:.3f} loss_domain_t={:.3f} loss_difference_s={:.3f} loss_difference_t={:.3f}'
            .format(dist.get_rank() + 1, epoch, step1_loss_epoch, step2_loss_epoch, cls_s_loss_epoch, cls_t_loss_epoch,
                    domain_s_loss_epoch, domain_t_loss_epoch, difference_s_loss_epoch, difference_t_loss_epoch))
        logging.info(
            'rank{} train epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                      Ps[c], Rs[c], F1S[c]))

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        domain_classifier.eval()
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss, confidence_sum = 0., 0.
        for x_t, label in val_bar:
            x_t, label = x_t.to(device), label.to(device)
            optimizer_model.zero_grad()
            optimizer_domain_classifier.zero_grad()
            shared_f_t, private_f_t, = model(x_t, 2)
            domain_out_t = domain_classifier(shared_f_t)[-1]
            scores_t = cal_grad_scores(shared_f_t, domain_out_t, 2)

            filter_num = int(model.module.filter_ratio * scores_t.size()[1])
            mask_t_ds = torch.ones_like(shared_f_t).to(device)
            row = torch.arange(mask_t_ds.size()[0]).unsqueeze(1).to(device)
            _, index_t_ds_channels = scores_t.topk(filter_num)
            mask_t_ds[row, index_t_ds_channels] = 0.
            masked_ds_shared_t_features = shared_f_t * mask_t_ds
            y_t = classifier(masked_ds_shared_t_features)[-1]

            cls_loss = val_criterion(y_s=y_t, label_s=label)
            val_loss += cls_loss.item()
            confidence, pseudo_labels = F.softmax(y_t.detach(), dim=1).max(dim=1)
            confidence_sum += sum(confidence)
            pred = y_t.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())
            val_bar.set_postfix({
                'epoch': epoch,
                'loss': f'{cls_loss.item():.3f}',
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}'
            })
        val_loss /= len(val_dataloader) * CFG.DATALOADER.BATCH_SIZE
        confidence_sum /= len(val_dataloader) * CFG.DATALOADER.BATCH_SIZE

        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('val/loss-epoch', val_loss, epoch)
            writer.add_scalar('val/PA-epoch', PA, epoch)
            writer.add_scalar('val/mPA-epoch', mPA, epoch)
            writer.add_scalar('val/KC-epoch', KC, epoch)
            writer.add_scalar('val/confidence-epoch', confidence_sum, epoch)
        if PA > best_PA:
            best_epoch = epoch

        logging.info('rank{} val epoch={} | loss={:.3f}'.format(dist.get_rank() + 1, epoch, val_loss))
        logging.info(
            'rank{} val epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} val epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                    Ps[c], Rs[c], F1S[c]))

        for s in [scheduler_model, scheduler_classifier, scheduler_domain_classifier]:
            if s is not None:
                try:
                    s.step()
                except TypeError:
                    s.step(val_loss)
        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'model': {
                    'state_dict': model.state_dict(),
                },
                'classifier': {
                    'state_dict': classifier.state_dict(),
                },
                'domain_classifier': {
                    'state_dict': domain_classifier.state_dict(),
                },
                'optimizer': {
                    'state_dict': optimizer_model.state_dict(),
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
