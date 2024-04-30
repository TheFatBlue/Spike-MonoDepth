import os
import json
import logging
import argparse
import bisect
from os.path import join

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset

from data_loader.SpikeMono import *
from trainer.spiket_trainer import SpikeTTrainer
from model.model import *
from model.S2DepthNet import S2DepthTransformerUNetConv
from model.loss import *
from model.metric import *
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop


logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(description='Spike Transformer')
parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
parser.add_argument('-f', '--datafolder', default=None, type=str, help='datafolder root path (default: None)')
parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
parser.add_argument('-i', '--initial_checkpoint', default=None, type=str, help='path to the checkpoint with which to initialize the model weights (default: None)')
parser.add_argument('--num_threads', type=int, help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size', type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank', type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url', type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1241')
parser.add_argument('--dist_backend', type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu', type=int,   help='GPU id to use.', default=0)
parser.add_argument('--multiprocessing_distributed', help='Use multi-processing distributed training to launch '
                                                    'N processes per node, which has N GPUs. This is the '
                                                    'fastest way to use PyTorch for either single node or '
                                                    'multi node data parallel training', action='store_true',)

args = parser.parse_args()

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

def remove_module_prefix(state_dict):
    """Remove the 'module.' prefix from each key in the state_dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # remove `module.` prefix
        new_state_dict[new_key] = value
    return new_state_dict

def main_worker(gpu, ngpus_per_node, args):
# def main_worker(config, resume, initial_checkpoint=None, DeviceIds=None):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    else:
        torch.cuda.set_device(args.gpu)
        # print(f"Training on GPU {args.gpu} ...")
        # dist.init_process_group(backend="nccl", init_method="env://", world_size=1, rank=0)
    
    config = args.config 
    resume = args.resume
    initial_checkpoint = args.initial_checkpoint

    train_logger = None

    assert (config['trainer']['sequence_length'] > 0)

    dataset_type, base_folder = {}, {}
    scene, side = {}, {}
    clip_distance = {}
    reg_factor = {}
    baseline = {}

    # this will raise an exception if the env variable is not set
    # preprocessed_datasets_folder = os.environ['PREPROCESSED_DATASETS_FOLDER']

    use_phased_arch = config['use_phased_arch']
    loss_composition = config['trainer']['loss_composition']
    loss_weights = config['trainer']['loss_weights']
    normalize = config['data_loader'].get('normalize', True)
    
    for split in ['train', 'validation']:
        
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = config['data_loader'][split]['base_folder']
        scene[split] = config['data_loader'][split]['scene']
        side[split] = config['data_loader'][split]['side']

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        try:
            baseline[split] = config['data_loader'][split]['baseline']
        except KeyError:
            baseline[split] = False

        try:
            reg_factor[split] = config['data_loader'][split]['reg_factor']
        except KeyError:
            reg_factor[split] = 5.7

    train_dataset = concatenate_subfolders(
        base_folder=join(args.datafolder, base_folder['train']),
        dataset_type=dataset_type['train'],
        scene=scene['train'],
        side=side['train'],
        transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0), 
                           RandomCrop(224)]),
        clip_distance=clip_distance['train'],
        normalize=normalize,
        reg_factor=reg_factor['train'],
    )

    # no data augmentation for validation set
    validation_dataset = concatenate_subfolders(
        base_folder=join(args.datafolder, base_folder['validation']),
        dataset_type=dataset_type['validation'],
        scene=scene['validation'],
        side=side['validation'],
        transform=CenterCrop(224),
        clip_distance=clip_distance['validation'],
        normalize=normalize,
        reg_factor=reg_factor['validation'],
    )

    # Set up data loaders
    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    data_loader = DataLoader(
        train_dataset,
        batch_size=int(config['data_loader']['batch_size']/ ngpus_per_node),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **kwargs
    )

    if args.distributed:
        validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    else:
        validation_sampler = None
        
    valid_data_loader = DataLoader(
        validation_dataset,
        batch_size=int(config['data_loader']['batch_size']/ ngpus_per_node),
        sampler=validation_sampler,
        shuffle=(validation_sampler is None),
        **kwargs
    )

    config['model']['gpu'] = args.gpu
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']

    torch.manual_seed(111)
    model = eval(config['arch'])(config['model'])
    model.summary()
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(config['data_loader']['batch_size'] / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False, find_unused_parameters=True)
        else:
            model.cuda()
            model = DataParallelModel(model, find_unused_parameters=True)
    else:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
        else:
            model.cuda()

        args.batch_size = int(config['data_loader']['batch_size'])

    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")
    

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        modified_state_dict = remove_module_prefix(checkpoint['state_dict'])
        # model.load_state_dict(checkpoint['state_dict'])
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)  # tag="events"
        model.load_state_dict(modified_state_dict)
        
        for name, param in model.named_parameters():
            if 'encoder' in name or 'resblocks' in name:
                param.requires_grad = False
        print("Encoder layers frozen")
        model.summary()
    
    cudnn.benchmark = True
    

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = SpikeTTrainer(
        model, args, loss, loss_params, metrics,
        resume=resume,
        config=config,
        train_sampler=train_sampler,
        data_loader=data_loader, 
        ngpus_per_node=ngpus_per_node,
        valid_data_loader=valid_data_loader,
        train_logger=train_logger
    )

    trainer.train()


def main():
    logger = logging.getLogger()

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None
    args.config = config

    if args.multiprocessing_distributed:
        print("---- Distributed Training ----")
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print("---- Single Training ----")
        main_worker(args.gpu, ngpus_per_node, args)


if __name__ == '__main__':

    main()
