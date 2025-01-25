import os
import torch
import random
import logging
import argparse
import numpy as np
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
    # number of GPUs totally, which equals to the number of processes
    args.path = os.path.join(args.path, str(args.seed))
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
    torch.autograd.set_detect_anomaly(True)
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
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    for k, v in model.items():
        if k == 'D':
            for k2, v2 in v.items():
                v2.to(device)
        else:
            v.to(device)

    # build criterion
    loss_names = CFG.CRITERION.ITEMS
    loss_weights = CFG.CRITERION.WEIGHTS
    assert len(loss_names) == len(loss_weights)
    cls_criterion = build_criterion(loss_names[0])
    cls_criterion.to(device)
    domain_criterion = build_criterion(loss_names[1])
    domain_criterion.to(device)
    cons_criterion = build_criterion(loss_names[2])
    cons_criterion.to(device)
    content_criterion = build_criterion(loss_names[3])
    content_criterion.to(device)
    style_criterion = build_criterion(loss_names[4])
    style_criterion.to(device)
    gen_criterion = build_criterion(loss_names[5])
    gen_criterion.to(device)
    recon_criterion = build_criterion(loss_names[6])
    recon_criterion.to(device)

    val_criterion = build_criterion('softmax+ce')
    val_criterion.to(device)
    # build metric
    metric = Metric(NUM_CLASSES)
    # build optimizer
    optimizer_E = build_optimizer(model['E'])
    optimizer_D = dict()
    for k, v in model['D'].items():
        optimizer_D[k] = build_optimizer(v)
    optimizer_G = build_optimizer(model['G'])
    optimizer_S = build_optimizer(model['S'])
    optimizer_T = build_optimizer(model['T'])
    # build scheduler
    scheduler_E = build_scheduler(optimizer_E)
    scheduler_D = dict()
    for k, v in optimizer_D.items():
        scheduler_D[k] = build_scheduler(v)
    scheduler_G = build_scheduler(optimizer_G)
    scheduler_S = build_scheduler(optimizer_S)
    scheduler_T = build_scheduler(optimizer_T)
    # DDP
    for k, v in model.items():
        if k != 'D':
            model[k] = DistributedDataParallel(v, find_unused_parameters=True, broadcast_buffers=False)
        else:
            for k2, v2 in v.items():
                v[k2] = DistributedDataParallel(v2, find_unused_parameters=True, broadcast_buffers=False)

    # train function
    def set_zero_grad():
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()
        optimizer_E.zero_grad()
        optimizer_T.zero_grad()
        for k, v in optimizer_D.items():
            v.zero_grad()

    def train_dis(model, datas):  # Train Discriminators (D)
        for k2, v2 in model['D'].items():
            v2.train()
        set_zero_grad()
        features, converted_imgs, D_outputs_fake, D_outputs_real = dict(), dict(), dict(), dict()
        # Real
        for k, v in datas.items():
            features[k] = model['E'](v)
            D_outputs_real[k] = model['D'][k](v)[-1]
        converts = model['S'].module.converts
        contents, styles = model['S'](features, converts)
        # Fake
        for convert in converts:
            source, target = convert.split('2')
            converted_imgs[convert] = model['G'](contents[convert], styles[target])
            D_outputs_fake[convert] = model['D'][target](converted_imgs[convert].detach())[-1]

        dis_loss_1 = sum(
            domain_criterion(y_s=D_outputs_real[dset], label_s=domain_label_1) * loss_weights[1]
            for dset in D_outputs_real.keys()
        )
        dis_loss_2 = sum(
            domain_criterion(y_s=D_outputs_fake[cv], label_s=domain_label_0) * loss_weights[1]
            for cv in D_outputs_fake.keys()
        )
        dis_loss = dis_loss_1 + dis_loss_2
        dis_loss.backward()
        for optimizer in optimizer_D.values():
            optimizer.step()

        for k2, v2 in model['D'].items():
            v2.eval()

    def train_task(model, datas, labels):  # Train Task Networks (T)
        set_zero_grad()
        model['T'].train()
        features, converted_imgs, pred = dict(), dict(), dict()
        converts = model['S'].module.converts
        with torch.no_grad():
            for k, v in datas.items():
                features[k] = model['E'](v)
            contents, styles = model['S'](features, converts)
            for convert in converts:
                source, target = convert.split('2')
                converted_imgs[convert] = model['G'](contents[convert], styles[target])

        for convert in converts:
            pred[convert] = model['T'](converted_imgs[convert])[-1]
            source, target = convert.split('2')
            pred[source] = model['T'](datas[source])[-1]

        cls_loss = 0.
        y_s = None
        for key in pred.keys():
            if '2' in key:
                source, target = key.split('2')
            else:
                source = key
                y_s = pred[key]
            cls_loss += cls_criterion(pred[key], labels[source])
        cls_loss *= loss_weights[0]
        cls_loss.backward()
        optimizer_T.step()
        model['T'].eval()
        return y_s

    def train_esg(model, datas):  # Train Encoder(E), Separator(S), Generator(G)
        model['E'].train()
        model['S'].train()
        model['G'].train()

        def gram(x):
            (b, c, h, w) = x.size()
            f = x.view(b, c, h * w)
            f_T = f.transpose(1, 2)
            G = f.bmm(f_T) / (c * w * h)
            return G

        set_zero_grad()
        features, converted_imgs, recon_imgs, D_outputs_fake = dict(), dict(), dict(), dict()
        features_converted, perceptual, style_gram = dict(), dict(), dict()
        perceptual_converted, style_gram_converted = dict(), dict()
        converts = model['S'].module.converts

        for k, v in datas.items():
            features[k] = model['E'](v)
            recon_imgs[k] = model['G'](features[k], 0)
            perceptual[k] = model['P'](v)
            style_gram[k] = gram(perceptual[k])
        contents, styles = model['S'](features, converts)
        for convert in converts:
            source, target = convert.split('2')
            converted_imgs[convert] = model['G'](contents[convert], styles[target])
            D_outputs_fake[convert] = model['D'][target](converted_imgs[convert])[-1]
            features_converted[convert] = model['E'](converted_imgs[convert])
            perceptual_converted[convert] = model['P'](converted_imgs[convert])
            style_gram_converted[convert] = gram(perceptual_converted[convert])
        contents_converted, styles_converted = model['S'](features_converted)

        # Content Loss
        Content_loss = 0.
        for cv in perceptual_converted.keys():
            source, target = cv.split('2')
            Content_loss += content_criterion(perceptual[source][-1], perceptual_converted[cv][-1])
        Content_loss *= loss_weights[3]
        # Style Loss
        Style_loss = 0.
        for cv in style_gram_converted.keys():
            source, target = cv.split('2')
            Style_loss += style_criterion(style_gram[target], style_gram_converted[cv])
        Style_loss *= loss_weights[4]
        # Consistency Loss
        Consistency_loss = 0.
        for cv in converts:
            source, target = cv.split('2')
            Consistency_loss += cons_criterion(contents[cv], contents_converted[cv])
            Consistency_loss += cons_criterion(styles[target], styles_converted[cv])
        Consistency_loss *= loss_weights[2]
        # Generator Loss
        G_loss = 0.
        for cv in D_outputs_fake.keys():
            G_loss += gen_criterion(D_outputs_fake[cv], domain_label_1)
        G_loss *= loss_weights[5]
        # Reconstruction Loss
        Recon_loss = 0.
        for dset in datas.keys():
            Recon_loss += recon_criterion(datas[dset], recon_imgs[dset])
        Recon_loss *= loss_weights[6]

        errESG = G_loss + Content_loss + Style_loss + Consistency_loss + Recon_loss
        errESG.backward()

        optimizer_E.step()
        optimizer_S.step()
        optimizer_G.step()
        model['E'].eval()
        model['S'].eval()
        model['G'].eval()


    epoch = 0
    iteration = 0
    best_epoch = 0
    best_PA = 0.

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
            lr = optimizer_E.param_groups[0]['lr']
            writer.add_scalar('lr-epoch', lr, epoch)

        # train
        # set_train(model)

        metric.reset()  # reset metric
        train_bar = tqdm(range(1, CFG.DATALOADER.ITERATION + 1), desc='training', ascii=True)
        for iteration in train_bar:
            x_s, label_s = next(source_iterator)
            x_t, _ = next(target_iterator)
            x_s, label_s = x_s.to(device), label_s.to(device)
            x_t = x_t.to(device)
            datas = {'S': x_s, 'T': x_t}
            labels = {'S': label_s}
            domain_label_0 = torch.zeros(len(label_s))
            domain_label_1 = torch.ones(len(label_s))
            domain_label_0, domain_label_1 = domain_label_0.to(device), domain_label_1.to(device)

            # training
            train_dis(model, datas)
            for t in range(2):
                train_esg(model, datas)
            y_s = train_task(model, datas, labels)

            pred = y_s.argmax(axis=1)
            metric.add(pred.data.cpu().numpy(), label_s.data.cpu().numpy())
            train_bar.set_postfix({
                'epoch': epoch,
                'mP': f'{metric.mPA():.3f}',
                'PA': f'{metric.PA():.3f}',
                'KC': f'{metric.KC():.3f}'
            })

        PA, mPA, Ps, Rs, F1S, KC = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s(), metric.KC()
        if dist.get_rank() == 0:
            writer.add_scalar('train/PA-epoch', PA, epoch)
            writer.add_scalar('train/mPA-epoch', mPA, epoch)
            writer.add_scalar('train/KC-epoch', KC, epoch)
        logging.info(
            'rank{} train epoch={} | '.format(
                dist.get_rank() + 1, epoch))
        logging.info(
            'rank{} train epoch={} | PA={:.3f} mPA={:.3f} KC={:.3f}'.format(dist.get_rank() + 1, epoch, PA, mPA, KC))
        for c in range(NUM_CLASSES):
            logging.info(
                'rank{} train epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                      Ps[c], Rs[c], F1S[c]))

        # validate
        if args.no_validate:
            continue
        # 由于 retain graph = true 此处不能用eval() 否则计算图会被free掉 导致模型失效
        metric.reset()  # reset metric
        val_bar = tqdm(val_dataloader, desc='validating', ascii=True)
        val_loss = 0.
        model['T'].eval()
        with torch.no_grad():  # disable gradient back-propagation
            for x_t, label_t in val_bar:
                x_t, label_t = x_t.to(device), label_t.to(device)
                with torch.no_grad():
                    y_t = model['T'](x_t)[-1]
                pred = y_t.argmax(axis=1)
                metric.add(pred.data.cpu().numpy(), label_t.data.cpu().numpy())
                val_bar.set_postfix({
                    'epoch': epoch,
                    'mP': f'{metric.mPA():.3f}',
                    'PA': f'{metric.PA():.3f}',
                    'KC': f'{metric.KC():.3f}'
                })
        val_loss /= len(val_dataloader) * CFG.DATALOADER.BATCH_SIZE

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
                'rank{} val epoch={} | class={} P={:.3f} R={:.3f} F1={:.3f}'.format(dist.get_rank() + 1, epoch, c,
                                                                                    Ps[c], Rs[c], F1S[c]))

        # adjust learning rate if specified
        for s in [scheduler_S, scheduler_T, scheduler_G, scheduler_E, *scheduler_D.values()]:
            if s is not None:
                try:
                    s.step()
                except TypeError:
                    s.step(val_loss)

        # save checkpoint
        if dist.get_rank() == 0:
            if PA > best_PA:
                best_PA = PA
                # torch.save(checkpoint, os.path.join(args.path, 'best.pth'))
            writer.add_scalar('best-PA', best_PA, epoch)


def main():
    # parse command line arguments
    args = parse_args()

    # multi processes, each process runs worker(i,args) i=range(nprocs)
    # total processes = world size = nprocs*nodes
    mp.spawn(worker, args=(args,), nprocs=args.gpus)


if __name__ == '__main__':
    main()
