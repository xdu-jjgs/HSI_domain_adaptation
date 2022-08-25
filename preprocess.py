import os
import argparse

import torch
from tqdm import tqdm
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

    splits = ['train', 'val', 'test']
    for index, split in enumerate(splits):
        save_path = os.path.join(args.path, split)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        dataset = build_dataset(split)
        dataset_bar = tqdm(range(len(dataset)), desc='preprocessing {} dataset'.format(split), ascii=True)
        for ind in dataset_bar:
            data, label = dataset[ind]
            print(data.shape, label)
            torch.save(data, os.path.join(save_path, '{}_{}.pt'.format(ind, label)))


if __name__ == '__main__':
    main()
