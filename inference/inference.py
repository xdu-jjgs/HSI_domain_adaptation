import os

import numpy as np
import torch
import logging
import argparse

from tqdm import tqdm
from datetime import datetime

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
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device for test')
    parser.add_argument('--path',
                        type=str,
                        default=os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S-test')),
                        help='path for experiment output files')
    args = parser.parse_args()
    return args


def main():
    # parse command line arguments
    args = parse_args()
    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    # merge config with config file
    CFG.merge_from_file(args.config)

    # dump config
    with open(os.path.join(args.path, 'config.yaml'), 'w') as f:
        f.write(CFG.dump())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.path, 'test.log')),
            logging.StreamHandler(),
        ])

    # build dataset
    class_interest = CFG.DATASET.CLASSES_INTEREST
    test_dataset = build_dataset('test')
    NUM_CHANNELS = test_dataset.num_channels
    NUM_CLASSES = len(class_interest)
    # build data loader
    test_dataloader = build_dataloader(test_dataset, 'test', drop_last=False)
    # build model
    model = build_model(NUM_CHANNELS, NUM_CLASSES)
    model.to(args.device)
    # build metric
    metric = Metric(NUM_CLASSES)

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise RuntimeError('checkpoint {} not found'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    # delete module saved in train
    model.load_state_dict(({k.replace('module.', ''): v for k, v in checkpoint['model']['state_dict'].items()}))
    best_PA = checkpoint['metric']['PA']
    logging.info('load checkpoint {} with PA={:.3f}'.format(args.checkpoint, best_PA))

    # inference
    model.eval()  # set model to evaluation mode
    metric.reset()  # reset metric
    test_bar = tqdm(test_dataloader, desc='testing', ascii=True)
    res = []
    with torch.no_grad():  # disable gradient back-propagation
        for batch, (x, label) in enumerate(test_bar):
            # change DoubleTensor x to FloatTensor
            x, label = x.float().to(args.device), label.to(args.device)
            y = model(x)

            if NUM_CLASSES > 2:
                pred = y.data.cpu().numpy().argmax(axis=1)
            else:
                pred = (y.data.cpu().numpy() > 0.5).squeeze(1)
            label = label.data.cpu().numpy()
            res.append(pred)
            metric.add(pred, label)
    res = np.concatenate(res)
    PA, mPA, Ps, Rs, F1S = metric.PA(), metric.mPA(), metric.Ps(), metric.Rs(), metric.F1s()
    logging.info('inference | PA={:.3f} mPA={:.3f} worst={:.3f}'.format(PA, mPA, min(Ps)))
    for c in range(NUM_CLASSES):
        logging.info(
            'inference | class={}-{} P={:.3f} R={:.3f} F1={:.3f}'.format(c, test_dataset.names[class_interest[c]], Ps[c],
                                                                         Rs[c], F1S[c]))
    # logging.info(metric.matrix)
    plot_confusion_matrix(metric.matrix, os.path.join(args.path, 'confusion_matrix.png'))

    plot_classification_image(test_dataset, res,os.path.join(args.path, 'classification_map.png'))

if __name__ == '__main__':
    main()