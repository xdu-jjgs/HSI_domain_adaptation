import os
import torch
import numpy as np
import argparse

from datetime import datetime

from configs import CFG
from datas import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config',
                        type=str,
                        help='config file')
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

    splits = ['train', 'test']
    save_path = os.path.join(args.path)
    for index, split in enumerate(splits):
        dataset = build_dataset(split)
        print("{} dataset data size {}".format(split, dataset.data.size()))
        print("{} dataset gt size {}".format(split, dataset.gt.shape))
        torch.save(dataset.data, os.path.join(save_path, '{}_data.pt'.format(split)), pickle_protocol=4)
        np.save(os.path.join(save_path, '{}_gt.npy'.format(split)), dataset.gt)


if __name__ == '__main__':
    main()
