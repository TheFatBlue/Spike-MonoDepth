import cv2
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset

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
    
def obtain_spike_video(spikes, video_filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')
    timestamps = spikes.shape[0]

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    for iSpk in range(timestamps):
        tmpSpk = spikes[iSpk, :, :] * 255
        tmpSpk = cv2.cvtColor(tmpSpk.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mov.write(tmpSpk)

    mov.release()
    
spike_obj = SpikeStream(filepath="dataset/Spike-Stero/train/0000/indoor/left/0001/0001.dat", spike_h=250, spike_w=400, print_dat_detail=False)
        
spike = spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
dataDict = {
    'spike_h': 250,  # 视频的高度
    'spike_w': 400   # 视频的宽度
}

# obtain_spike_video(spike, "viz_output/il_0001.avi", **dataDict)

label = np.load("dataset/Spike-Mini/train/0000/indoor/left/0000/0000_gt.npy").astype(np.float32)
label = np.clip(label, 1e-3, 4.0) / 4.0
# print(np.max(label), np.min(label))
label = 1.0 + np.log(label) / 1.86

label = label.clip(0, 1.0)
# print(np.max(label)-np.min(label))

if len(label.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
    label = np.expand_dims(label, -1)
label = np.moveaxis(label, -1, 0)  # H x W x 1 -> 1 x H x W

print(label[0, :45, 310:].mean(), label[0, 205:, 310:].mean())

import sys

# sys.exit()

label = np.rot90(label, 2)  # 将图像旋转180度


arr_scaled = (label[0] * 255).astype(np.uint8)

# 保存图像
cv2.imwrite('output.png', arr_scaled)