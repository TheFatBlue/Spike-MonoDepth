import os
import json
import logging
import argparse
import sys
import bisect
from os.path import join

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from model.model import *
from model.loss import *
from model.metric import *
from model.S2DepthNet import S2DepthTransformerUNetConv
from data_loader.SpikeMono import *
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop


class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


def concatenate_subfolders(base_folder,
                           dataset_type,
                           scene='indoor',
                           side='left',
                           transform=None,
                           clip_distance=100.0,
                           normalize=True,
                           reg_factor=5.7,
                           dataset_idx_flag=False):
    """
    Create an instance of ConcatDataset by aggregating all the datasets in a given folder
    """
    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))

    args_dict = {
        'base_folder': '',
        'scene': scene,
        'side': side,
        'transform': transform,
        'clip_distance': clip_distance,
        'normalize': normalize,
        'reg_factor': reg_factor
    }
    
    datasets = []
    for subfolder in subfolders:
        args_dict['base_folder'] = join(base_folder, subfolder)
        datasets.append(eval(dataset_type)(**args_dict))

    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(datasets)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(datasets)

    return concat_dataset


logging.basicConfig(level=logging.INFO, format='')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from each key in the state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # remove `module.` prefix
        new_state_dict[new_key] = value
    return new_state_dict

def main(config, initial_checkpoint, data_folder):
    train_logger = None

    calculate_scale = True

    total_metrics = []
    
    every_x_rgb_frame = {}
    every_x_rgb_frame['validation'] = 1

    # this will raise an exception is the env variable is not set
    # preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    loss_composition = config['trainer']['loss_composition']
    normalize = config['data_loader'].get('normalize', True)

    dataset_type = config['data_loader']['validation']['type']
    # base_folder['validation'] = data_folder
    scene = config['data_loader']['validation']['scene']
    side = config['data_loader']['validation']['side']

    try:
        clip_distance = config['data_loader']['validation']['clip_distance']
    except KeyError:
        clip_distance = 100.0

    try:
        baseline = config['data_loader']['validation']['baseline']
    except KeyError:
        baseline = False

    try:
        reg_factor = config['data_loader']['validation']['reg_factor']
    except KeyError:
        reg_factor = 5.7

    test_dataset = concatenate_subfolders(
        base_folder=data_folder,
        dataset_type=dataset_type,
        scene=scene,
        side=side,
        transform=CenterCrop(224),
        clip_distance=clip_distance,
        normalize=normalize,
        reg_factor=reg_factor,
        dataset_idx_flag=True,
    )

    config['model']['gpu'] = config['gpu']
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']
    
    model = eval(config['arch'])(config['model'])
    # model.summary()

    print('Loading initial model weights from: {}'.format(initial_checkpoint))
    checkpoint = torch.load(initial_checkpoint)
    modified_state_dict = remove_module_prefix(checkpoint['state_dict'])
    model.load_state_dict(modified_state_dict, strict=False)

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    model.eval()

    with torch.no_grad():
        
        item, dataset_idx = test_dataset[0]
        input = {}
        for key, value in item[0].items():
            input[key] = value[None, :]
        predicted_targets, _, _ = model(input, None, 0)
        

if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Spike Transformer')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)


    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model, args.data_folder)
