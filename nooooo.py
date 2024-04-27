import numpy as np

path = "dataset/Spike-Stero/train/0000/indoor/left/0010/0010_gt.npy"
label = np.load(path).astype(np.float32)
label = np.clip(label, 0.0, 1000.0) / 1000.0

print(np.max(label), np.min(label))