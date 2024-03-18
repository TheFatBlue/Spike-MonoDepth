import numpy as np
import os
import shutil
from data_loader.spike_stero import DatasetSpikeStero_Mono  # 确保这里的路径正确

# 步骤1: 创建模拟数据环境
def create_mock_environment(root_dir='mock_data'):
    os.makedirs(root_dir, exist_ok=True)
    scences = ['indoor', 'outdoor']
    for scence in scences:
        for side in ['left', 'right']:
            path = os.path.join(root_dir, scence, side, 'session1', 'example1')
            os.makedirs(path, exist_ok=True)
            # 创建模拟数据文件
            np.save(os.path.join(path, 'data.npy'), np.random.rand(250, 400))  # 随机生成的数据
            # 创建模拟标签文件
            if side == 'left':
                label_dir = os.path.join(root_dir, scence, side, 'session1', 'example1', 'labels')
                os.makedirs(label_dir, exist_ok=True)
                np.save(os.path.join(label_dir, 'label.npy'), np.random.rand(250, 400))

# 步骤2: 定义一个模拟的data_parameter_dict函数
def mock_data_parameter_dict(filepath, task):
    return {
        'filepath': filepath,
        'labeled_data_dir': os.path.join(os.path.dirname(filepath), 'labels', 'label.npy')
    }

# 步骤3: 测试DatasetSpikeStero_Mono类
def test_dataset_spike_stero_mono():
    # create_mock_environment()
    
    # 替换DatasetSpikeStero_Mono中的data_parameter_dict为我们的模拟函数
    # global data_parameter_dict
    # data_parameter_dict = mock_data_parameter_dict
    
    dataset = DatasetSpikeStero_Mono(filepath='dataset/Spike-Stero/', split='train', scence='indoor')
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(f"Sample {i}: spike shape {sample['spike'].shape}, depth shape {sample['depth'].shape}")
    print(len(dataset))
    sample = dataset[0]
    print(sample['spike'].shape)
    
    # 清理模拟环境
    # shutil.rmtree('mock_data')

# 运行测试函数
test_dataset_spike_stero_mono()
