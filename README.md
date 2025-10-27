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

## Project Structure

```
Spike-MonoDepth/
├── configs/                   # Configuration files
├── data/                      # Dataset directory
│   └── Spike-Stereo/         # PKU-Spike-Stereo dataset
├── docs/                      # Documentation
├── experiments/               # Training outputs
│   ├── checkpoints/          # Model checkpoints
│   ├── logs/                 # Training logs
│   ├── metrics/              # Evaluation metrics
│   └── results/              # Result visualizations
├── scripts/                   # Executable scripts
│   ├── train.py              # Training script
│   ├── test_DENSE.py         # Testing script
│   ├── evaluation_DENSE.py   # Evaluation script
│   └── *.sh                  # Shell scripts
├── src/spike_monodepth/      # Main source code
│   ├── base/                 # Base classes
│   ├── data/                 # Data loaders
│   ├── losses/               # Loss functions
│   ├── metrics/              # Evaluation metrics
│   ├── models/               # Model architectures
│   ├── trainers/             # Training logic
│   └── utils/                # Utility functions
├── tests/                     # Unit tests
├── README.md
├── requirements.txt
└── setup.py                   # Package setup
```

## Installation

### Option 1: Install as Package (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/TheFatBlue/Spike-MonoDepth.git
   cd Spike-MonoDepth
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

### Option 2: Manual Installation

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

### Training

To train the model on the PKU-Spike-Stereo dataset:

```bash
cd scripts
python train.py --config ../configs/ft_il_100e.json \
                --datafolder ../data \
                --gpu 0
```

Or use the provided shell script:

```bash
cd scripts
bash run.sh
```

### Testing

To test the model:

```bash
cd scripts
python test_DENSE.py \
    --path_to_model ../experiments/checkpoints/model_best.pth.tar \
    --output_path ../experiments/results \
    --data_folder ../data/Spike-Stereo/test
```

### Evaluation

To evaluate model predictions:

```bash
cd scripts
python evaluation_DENSE.py \
    --target_dataset ../experiments/results/ground_truth/npy/depth_image/ \
    --predictions_dataset ../experiments/results/npy/image/ \
    --output_folder ../experiments/metrics
```

## Results

We achieved significant improvements in depth prediction accuracy compared to traditional models when dealing with neuromorphic camera data. For detailed experimental results and comparison, refer to the [project report](https://github.com/TheFatBlue/Spike-MonoDepth/docs/report.pdf).

## Dataset

We used the **PKU-Spike-Stereo** dataset for this project. It contains pulse streams and corresponding depth maps, captured in both indoor and outdoor scenes. The dataset can be downloaded from [here](https://openi.pcl.ac.cn/Cordium/SpikeCV/datasets).

## Contact

For any questions or collaboration inquiries, feel free to contact me at [thefatblue@gmail.com](mailto:thefatblue@gmail.com).
