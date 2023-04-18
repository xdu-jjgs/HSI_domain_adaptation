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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from configs import CFG
from metric import Metric
from models import build_model, DQN
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
    args.path = os.path.join(args.path, str(args.seed))
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
    selected_dataset = build_dataset('dynamic')
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
    target_dataloader = build_dataloader(target_dataset, sampler=target_sampler)
    val_dataloader = build_dataloader(val_dataset, sampler=val_sampler)

    # build data iteration
    source_iterator = build_iterator(source_dataloader)
    target_iterator = build_iterator(target_dataloader)
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(device)
    # todo: update initiation
    dqn = DQN(len_states=64 * CFG.DATALOADER.BATCH_SIZE, num_actions=CFG.DATALOADER.BATCH_SIZE,
              batch_size=CFG.DATALOADER.BATCH_SIZE)
    dqn.to(device)
    # print(model)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    dqn_criterion = build_criterion('l2dis')
    dqn_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    metric_pse = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_da = build_optimizer(model)
    optimizer_dqn = build_optimizer(dqn)
    # build scheduler
    scheduler1 = build_scheduler(optimizer_da)
    scheduler2 = build_scheduler(optimizer_dqn)

    # mixed precision
    [model, dqn], [optimizer_da, optimizer_dqn] = amp.initialize([model, dqn], [optimizer_da, optimizer_dqn],
                                                                 opt_level=args.opt_level)
    # DDP
    model = DistributedDataParallel(model, broadcast_buffers=False)
    # DistributedDataParallel不能有新方法？
    dqn = DistributedDataParallel(dqn)

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
        dqn.load_state_dict(checkpoint['dqn']['state_dict'])
        optimizer_da.load_state_dict(checkpoint['optimizer_da']['state_dict'])
        optimizer_dqn.load_state_dict(checkpoint['optimizer_dqn']['state_dict'])
        epoch = checkpoint['optimizer_da']['epoch']
        iteration = checkpoint['optimizer_da']['iteration']
        best_PA = checkpoint['metric']['PA']
        best_epoch = checkpoint['optimizer_da']['best_epoch']
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
            lr = optimizer_da.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train da model
        model.train()  # set model to training mode
        metric.reset()  # reset metric
        metric_pse.reset()
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        total_loss_epoch, cls_loss_epoch, sel_loss_epoch = 0., 0., 0.
        for iteration in train_bar:
            x_s, label = next(source_iterator)
            x_t, label_t = next(target_iterator)
            x_s, label = x_s.to(device), label.to(device)
            x_t = x_t.to(device)

            f_s, y_s = model(x_s)

            model.eval()
            with torch.no_grad():
                f_t, y_t = model(x_t)
            model.train()

            # DQN select
            dqn.train()
            metric_pse.reset()
            selected_dataset.flush()
            confidence, pseudo_labels = F.softmax(y_t.detach(), dim=1).max(dim=1)
            state = f_t.detach()
            dqn.module.current_state = state
            # mask = (confidence > CFG.CRITERION.THRESHOLD).float()
            # valid_index = torch.where(mask)
            # x_t_select = x_t[valid_index]
            # pseudo_labels_select = pseudo_labels[valid_index]
            # confidence_select = confidence[valid_index]
            # 先补齐到batch_size大小再选
            # state = get_next_state(state, CFG.DATALOADER.BATCH_SIZE-valid_index[0].size()[0])

            num_select = 0
            while state.size()[0] > 0:
                terminal = 0
                # action_index在dqn中已经减去了select_iteration，不会选到空样本
                action, action_index = dqn.module.choose_action(num_select)
                # TODO: update reward
                r_threshold = 0.8
                if confidence[action_index] > r_threshold:
                    reward = 1.
                elif num_select <= 1:
                    reward = 0.1
                else:
                    reward = -1

                if dqn.module.step < dqn.module.step_observe or reward > 0:
                    selected_dataset.append(x_t[action_index], pseudo_labels[action_index])
                    metric_pse.add(pseudo_labels[action_index].cpu().numpy(), label_t[action_index].cpu().numpy())

                x_t = torch.concat([x_t[:action_index], x_t[action_index + 1:]])
                state = torch.concat([state[:action_index], state[action_index + 1:]])
                pseudo_labels = torch.concat([pseudo_labels[:action_index], pseudo_labels[action_index + 1:]])
                confidence = torch.concat([confidence[:action_index], confidence[action_index + 1:]])
                label_t = torch.concat([label_t[:action_index], label_t[action_index + 1:]])

                num_select += 1
                # make next state
                feature_dim = state.size()[-1]
                zero_padding = torch.zeros([num_select, feature_dim]).to(device)
                next_state = torch.cat([state, zero_padding])
                # next_state = get_next_state(state, num_select, device)
                if reward < 0:
                    terminal = 1
                dqn.module.store_transition(action, reward, next_state, num_select, terminal, iteration)
                if dqn.module.step > dqn.module.step_observe and terminal == 1:
                    q_eval, q_target = dqn.module.train_net()
                    q_loss = dqn_criterion(q_eval, q_target)
                    optimizer_dqn.zero_grad()
                    with amp.scale_loss(q_loss, optimizer_dqn) as scaled_loss:
                        scaled_loss.backward()
                    optimizer_dqn.step()
                if reward < 0:
                    break

            # train da model by selected data
            model.train()
            selected_sampler = DistributedSampler(selected_dataset, shuffle=True)
            selected_dataloder = build_dataloader(selected_dataset, sampler=selected_sampler, drop_last=False)
            assert len(selected_dataloder) == 1
            x_st, pseudo_labels_st = next(iter(selected_dataloder))
            # print(x_st.shape, pseudo_labels_st.shape)
            f_st, y_st = model(x_st)

            cls_loss = cls_criterion(label_s=label, y_s=y_s) * loss_weights[0]
            sel_loss = cls_criterion(y_s=y_st, label_s=pseudo_labels_st) * loss_weights[1]
            total_loss = cls_loss + sel_loss

            cls_loss_epoch += cls_loss.item()
            sel_loss_epoch += sel_loss.item()
            total_loss_epoch += total_loss.item()

            optimizer_da.zero_grad()
            with amp.scale_loss(total_loss, optimizer_da) as scaled_loss:
                scaled_loss.backward()
            optimizer_da.step()

            pred = y_s.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label.data.cpu().numpy())

            train_bar.set_postfix({
                'iteration': iteration,
                'epoch': epoch,
                'loss': f'{total_loss.item():.3f}',
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}',
            })

        total_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        cls_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        sel_loss_epoch /= iteration * CFG.DATALOADER.BATCH_SIZE
        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        PA_pse, mPA_pse, Ps_pse, Rs_pse, F1S_pse, KC_pse = \
            metric_pse.PA(), metric_pse.mPA(), metric_pse.Ps(), metric_pse.Rs(), metric_pse.F1s(), metric_pse.KC()

        if dist.get_rank() == 0:
            writer.add_scalar('train/loss_total-epoch', total_loss_epoch, epoch)
            writer.add_scalar('train/loss_cls-epoch', cls_loss_epoch, epoch)
            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)
            writer.add_scalar('dqn/PA-epoch', PA_pse, epoch)
            writer.add_scalar('dqn/mPA-epoch', mPA_pse, epoch)
            writer.add_scalar('dqn/KC-epoch', KC_pse, epoch)
        logging.info(
            'rank{} train epoch={} | loss_total={:.3f} loss_cls={:.3f} loss_sel={:.3f}'.format(
                dist.get_rank() + 1, epoch, total_loss_epoch, cls_loss_epoch, sel_loss_epoch))
        logging.info(
            'rank{} train epoch={} | '
            'PA={:.3f} mPA={:.3f} KC={:.3f} | '
            'pse PA={:.3f} mPA={:.3f} KC={:.3f} NUM={}/{}'.format(
                dist.get_rank() + 1, epoch, PA, mPA, KC, PA_pse, mPA_pse, KC_pse, metric_pse.count,
                iteration * CFG.DATALOADER.BATCH_SIZE))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}| pse P={:.3f} R={:.3f} F1={:.3f}'.format(
                    dist.get_rank() + 1, epoch, c, Ps[c], Rs[c], F1S[c], Ps_pse[c], Rs_pse[c], F1S_pse[c]))

        # validate
        if args.no_validate:
            continue
        model.eval()  # set model to evaluation mode
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label in val_bar:
                x_t, label = x_t.to(device), label.to(device)
                _, y_t = model(x_t)

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

        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
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
        if scheduler1 is not None:
            try:
                scheduler1.step()
            except TypeError:
                scheduler1.step(val_loss)

        # adjust learning rate if specified
        if scheduler2 is not None:
            try:
                scheduler2.step()
            except TypeError:
                scheduler2.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            checkpoint = {
                'model': {
                    'state_dict': model.state_dict(),
                },
                'optimizer_da': {
                    'state_dict': optimizer_da.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'iteration': iteration
                },
                'dqn': {
                    'state_dict': dqn.state_dict(),
                },
                'optimizer_dqn': {
                    'state_dict': optimizer_dqn.state_dict(),
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
