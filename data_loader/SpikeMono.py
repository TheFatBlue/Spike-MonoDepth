# -*- encoding: utf-8 -*-

# here put the import lib
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class SpikeMonoDataset(Dataset):
    """Load sequences of time-synchronized {spike tensors + depth} from a folder."""

    def __init__(self, **kwargs):
        '''
        arg_keys:
            base_folder, scene, side,
            spike_h, spike_w,
            transform, clip_distance, normalize, reg_factor
        '''
        
        super().__init__()
        
        self.base_folder = kwargs.get('base_folder')
        
        self.scene = kwargs.get('scene')
        assert self.scene in ['indoor', 'outdoor', 'both'], "Invalid option, \'scene\' should be \'indoor\' or \'outdoor\'."
        
        self.side = kwargs.get('side')
        assert self.side in ['left', 'right'], "Invalid option, \'side\' should be in \'left\', \'right\'."

        self.spike_h = kwargs.get('spike_h', 250)
        self.spike_w = kwargs.get('spike_w', 400)
        
        self.transform = kwargs.get('transform', None)
        
        self.clip_distance = kwargs.get('clip_distance', 100.0)
        assert(self.clip_distance > 0)
        
        self.normalize = kwargs.get('normalize', True)
        
        self.reg_factor = kwargs.get('reg_factor', 5.7)

        self.path_list = self.__gen_data_list()
        self.length = len(self.path_list) * 3

    def __gen_data_list(self):
        path_list = []

        rootfolder = os.path.join(self.base_folder, self.scene, self.side)
        
        folders = sorted(os.listdir(rootfolder))
        for folder in folders:
            path_list.append(os.path.join(rootfolder, folder))

        return path_list
    
    def __len__(self):
        return self.length

    def __getitem__(self, i):
        assert(i >= 0)
        assert(i < self.length)

        seed = random.randint(0, 2**32)

        # the channel nums of the model is 128, but that of the dataset is 400, 
        # temporarily, I split each data evenly into 3 parts.
        index = i // 3
        bias = (i % 3) * 133

        path = self.path_list[index]

        path_dict = {
            'label': os.path.join(path, "{}_gt.npy".format(path.split('/')[-1])),
            'spike': os.path.join(path, "{}.dat".format(path.split('/')[-1]))
        }
        
        label = np.load(path_dict['label']).astype(np.float32)
        
        # clip and normalize
        label = np.clip(label, 0.0, self.clip_distance) / self.clip_distance
        label = 1.0 + np.log(label) / self.reg_factor
        label = label.clip(0, 1.0)
        
        if len(label.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            label = np.expand_dims(label, -1)
        label = np.moveaxis(label, -1, 0)  # H x W x 1 -> 1 x H x W
        label = torch.from_numpy(label)  # numpy to tensor

        if self.transform:
            random.seed(seed)
            label = self.transform(label)

        spike_obj = SpikeStream(filepath=path_dict['spike'], spike_h=250, spike_w=400, print_dat_detail=False)
        
        spike = spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
        spike = spike[bias:bias+128, :, :]
        
        if self.normalize:
            # normalize the spike tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
            # in the tensor are equal to (0.0, 1.0)
            mask = np.nonzero(spike)
            if mask[0].size > 0:
                mean, stddev = spike[mask].mean(), spike[mask].std()
                if stddev > 0:
                    spike[mask] = (spike[mask] - mean) / stddev

        spike = torch.from_numpy(spike)  # [C x H x W]
        if self.transform:
            random.seed(seed)
            spike = self.transform(spike)
            
        sequence = [{'image': spike,
                     'depth_image': label}]
        
        return sequence
    
class SpikeStream:

    def __init__(self, offline=True, camera_type=None, **kwargs):
        self.SpikeMatrix = None
        self.offline = True
        self.filename = kwargs.get('filepath')
        self.spike_width = kwargs.get('spike_w')
        self.spike_height = kwargs.get('spike_h')
        self.print_dat_detail = kwargs.get('print_dat_detail', False)

        self.camera_type = 1

    # return all spikes from dat file
    def get_spike_matrix(self, flipud=True, with_head=False):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.byte)
        if self.print_dat_detail:
            print(video_seq)
        img_size = self.spike_height * self.spike_width
        img_num = len(video_seq) // (img_size // 8)

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, total timestamp: %d' %
                  (self.spike_width, self.spike_height, img_num))

        # SpikeMatrix = np.zeros([img_num, self.spike_height, self.spike_width], np.byte)

        pix_id = np.arange(0, img_num * self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (img_num, self.spike_height, self.spike_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        data = video_seq[byte_id]
        result = np.bitwise_and(data, comparator)
        tmp_matrix = (result == comparator)
        # if with head, delete them
        if with_head:
            delete_indx = self.get_row_index()
            tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)

        if flipud:
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]
        else:
            self.SpikeMatrix = tmp_matrix

        file_reader.close()

        # self.SpikeMatrix = SpikeMatrix
        return self.SpikeMatrix