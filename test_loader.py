import os
import json
import logging
import argparse
import bisect
import sys
from os.path import join

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

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

config = json.load(open('configs/ft_ib_100e.json'))

normalize = config['data_loader'].get('normalize', True)

dataset_type = config['data_loader']['validation']['type']
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
    base_folder="dataset/Spike-Stero/test",
    dataset_type=dataset_type,
    scene=scene,
    side=side,
    transform=CenterCrop(224),
    clip_distance=clip_distance,
    normalize=normalize,
    reg_factor=reg_factor,
    dataset_idx_flag=True,
)

N = len(test_dataset)
print(f'N={N}')

idx = 0
prev_dataset_idx = -1
while idx < N:
    item, dataset_idx = test_dataset[idx]
    # print(item)
    # fuck = input()
    print(f'dataset_idx={dataset_idx}, idx={idx}, prev_dataset_idx={prev_dataset_idx}')
    if dataset_idx > prev_dataset_idx:
        print("hello")
        sequence_idx = 0
    if idx > 20 and idx < 24:
        print("test preview of index ", idx)
    if sequence_idx > 1:
        print(233)
        if idx % 100 == 0:
            print("saved image ", idx)
    sequence_idx += 1
    prev_dataset_idx = dataset_idx
    idx += 1
    print(f'sequence_idx={sequence_idx}')