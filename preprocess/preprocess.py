import os
import torch
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

    splits = ['train', 'val', 'test']
    save_path = os.path.join(args.path)
    for index, split in enumerate(splits):
        dataset = build_dataset(split)
        torch.save(dataset.data, os.path.join(save_path, '{}_data.pt'.format(split)), pickle_protocol=4)
        torch.save(dataset.gt, os.path.join(save_path, '{}_gt.pt'.format(split)))


if __name__ == '__main__':
    main()
