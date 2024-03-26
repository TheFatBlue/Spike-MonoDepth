from data_loader.SpikeMono import SpikeStream
import os
import numpy as np
from PIL import Image

root_path = 'dataset/Spike-Stero/train/0000/indoor/left'

for i in range(10):
    gt_path = os.path.join(root_path, '000'+str(i), '000'+str(i)+'_gt.npy')
    gt = np.load(gt_path).astype(np.float32)
    # print(gt.max(), gt.min())
    norm_gt = (gt - 0.8) * 255
    norm_gt = norm_gt.astype('uint8')

    img = Image.fromarray(norm_gt, 'L')

    img.save('tmp_out/' + str(i) + '.png')

