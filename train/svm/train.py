import os
import torch
import random
import logging
import argparse
import numpy as np

from datetime import datetime
from sklearn import svm
from sklearn.metrics import classification_report

from configs import CFG
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

    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    # number of GPUs totally, which equals to the number of processes
    args.path = os.path.join(args.path, str(args.seed))
    return args


def worker(args):
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

    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build dataset
    source_dataset = build_dataset('train')
    target_dataset = build_dataset('test')
    val_dataset = target_dataset
    assert source_dataset.num_classes == val_dataset.num_classes
    logging.info(
        "Number of train {}, val {}, test {}".format(len(source_dataset), len(val_dataset), len(target_dataset)))
    NUM_CLASSES = source_dataset.num_classes
    logging.info("Number of class: {}".format(NUM_CLASSES))
    # build data loader
    source_dataloader = build_dataloader(source_dataset, drop_last=False)
    val_dataloader = build_dataloader(val_dataset, drop_last=False)
    # build model
    model = svm.SVC()
    # print(model)

    # train - validation loop
    train_features = []
    train_labels = []
    for features, labels in source_dataloader:
        bs = features.shape[0]
        train_features.append(features.numpy().reshape(bs, -1))
        train_labels.append(labels)
    train_features = np.concatenate(train_features)[args.seed::50]
    train_labels = np.concatenate(train_labels)[args.seed::50]
    # print(train_features.shape)
    # print(train_labels.shape)
    model.fit(train_features, train_labels)

    test_features = []
    test_labels = []
    for features, labels in val_dataloader:
        bs = features.shape[0]
        test_features.append(features.numpy().reshape(bs, -1))
        test_labels.append(labels)
    test_features = np.concatenate(test_features)[::50]
    test_labels = np.concatenate(test_labels)[::50]
    predicted_labels = model.predict(test_features)

    print(classification_report(test_labels, predicted_labels))


def main():
    # parse command line arguments
    args = parse_args()
    worker(args)


if __name__ == '__main__':
    main()
