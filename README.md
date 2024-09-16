Here’s an English README for your GitHub repository:

---

# Spike-MonoDepth: Monocular Depth Estimation for Neuromorphic Cameras

This repository contains the implementation of the project *Monocular Depth Estimation for Neuromorphic Cameras*, which explores the use of neuromorphic spike cameras to achieve high-precision depth estimation for dynamic scenes.

## Overview

Neuromorphic cameras, unlike traditional cameras, capture visual information asynchronously with high temporal resolution, making them particularly well-suited for dynamic environments. However, due to the irregularity of the spike data they generate, traditional depth estimation models struggle to process this information effectively.

In this project, we introduce a novel depth estimation framework based on the **Spike Transformer**, designed specifically to handle the asynchronous pulse streams from neuromorphic cameras. Our method shows significant improvements in depth prediction accuracy and efficiency compared to conventional models when dealing with irregular data.

## Key Features

- **Spike Transformer**: A specialized encoder designed to process irregular pulse data from neuromorphic cameras.
- **Improved Decoder**: Enhanced architecture to improve prediction accuracy in depth estimation.
- **PKU-Spike-Stereo Dataset**: Trained and validated on the PKU-Spike-Stereo dataset, which includes synchronized pulse data and depth maps.
- **Applications**: Focuses on advancing depth estimation techniques in areas such as autonomous driving, robotics, and high-speed monitoring.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/TheFatBlue/Spike-MonoDepth.git
   cd Spike-MonoDepth
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you wish to use a GPU for faster training, ensure you have CUDA installed and that your environment is configured to utilize it.

## Usage

To train the model on the PKU-Spike-Stereo dataset:

1. Download the dataset and place it in the `data/` directory.

2. Start the training:
   ```bash
   python train.py --dataset_dir data/PKU-Spike-Stereo --batch_size 16 --epochs 100
   ```

3. For evaluation, run:
   ```bash
   python evaluate.py --model_path checkpoints/model_best.pth --dataset_dir data/PKU-Spike-Stereo
   ```

## Results

We achieved significant improvements in depth prediction accuracy compared to traditional models when dealing with neuromorphic camera data. For detailed experimental results and comparison, refer to the [project report](https://github.com/TheFatBlue/Spike-MonoDepth/docs/report.pdf).

## Dataset

We used the **PKU-Spike-Stereo** dataset for this project. It contains pulse streams and corresponding depth maps, captured in both indoor and outdoor scenes. The dataset can be downloaded from [here](https://openi.pcl.ac.cn/Cordium/SpikeCV/datasets).

## Contact

For any questions or collaboration inquiries, feel free to contact me at [thefatblue@gmail.com](mailto:thefatblue@gmail.com).
