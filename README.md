# Unlocking Generalization Power in LiDAR Point Cloud Registration (CVPR 2025)

In real-world environments, a LiDAR point cloud registration method with robust generalization capabilities (across varying distances and datasets) is crucial for ensuring safety in autonomous driving and other LiDAR-based applications. However, current methods fall short in achieving this level of generalization. To address these limitations, we propose UGP, a pruned framework designed to enhance generalization power for LiDAR point cloud registration. The core insight in UGP is the elimination of cross-attention mechanisms to improve generalization, allowing the network to concentrate on intra-frame feature extraction. Additionally, we introduce a progressive self-attention module to reduce ambiguity in large-scale scenes and integrate Bird’s Eye View (BEV) features to incorporate semantic information about scene elements. Together, these enhancements significantly boost the network’s generalization performance. We validated our approach through various generalization experiments in multiple outdoor scenes. In cross-distance generalization experiments on KITTI and nuScenes, UGP achieved state-of-the-art mean Registration Recall rates of 94.5\% and 91.4\%, respectively. In cross-dataset generalization from nuScenes to KITTI, UGP achieved a state-of-the-art mean Registration Recall of 90.9\%.

## News

20250405 - Our paper has been selected as a CVPR'25 Highlight!

20250313 - Our paper is now available on [ArXiv](https://arxiv.org/abs/2503.10149)!

20250227 - Our paper has been accepted by CVPR'25!

## Motivation

<div align="center">
<img src=assets\motivation.png>
</div>

## Overview of UGP

<div align="center">
<img src=assets\pipeline.png>
</div>

## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n UGP python==3.8
conda activate UGP

# [Optional] If you are using CUDA 11.0 or newer, please install `torch==1.7.1+cu110`
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```

## Pre-trained Models
```
mkdir ckpts
```
We provide pre-trained models in [Google Drive](https://drive.google.com/drive/folders/1XbwTou-bINwVy1RtdxNSUJ16Qlf-uPTw?usp=drive_link). Please download the latest weights and place them in the `ckpts` directory.

## Data Preparation

We provide two ways to prepare the data for training and evaluation:

### Option 1: Download our processed data directly

You can also directly download our processed KITTI / nuScenes data from Google Drive.

### Option 2: Download from the official websites

#### KITTI

Please download the KITTI Odometry dataset from the [KITTI official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), place it under `data/Kitti`, and run `data/Kitti/downsample_pcd.py` to generate the downsampled point clouds.


#### nuScenes
Please follow the [nuScenes official website](https://www.nuscenes.org/nuscenes#download) to download the lidar blobs (under file blobs) and metadata for the trainval and test splits in the Full dataset (v1.0) section. Only LiDAR scans and pose annotations are used.

The original nuScenes data organization is not well suited for point cloud registration tasks, so we convert the LiDAR data into KITTI format for easier development and extension. Thanks to the tools provided by [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit), this conversion requires only minimal modifications. Please place the downloaded nuScenes data under `data/nuscenes` and run `data/nuScenes/downsample_pcd.py` to generate the downsampled point clouds.


## Training

Use the following commands for training.

```bash
cd ./experiments/UGP.kitti
#or
cd ./experiments/UGP.nuscenes

# Single-GPU training
CUDA_VISIBLE_DEVICES=0 python trainval.py

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 trainval.py
```


## Testing

Use the following commands for testing.

```bash
# Run inference with a trained checkpoint
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=/Code/UGP/ckpts/XXX.pth.tar --distance=10 # 20, 30, 40

# Evaluate with the LGR estimator
CUDA_VISIBLE_DEVICES=0 python eval.py --method=lgr

# Evaluate with the RANSAC estimator
CUDA_VISIBLE_DEVICES=0 python eval.py --method=ransac --num_corr=50000
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=peakpang/UGP&type=Date)](https://star-history.com/#peakpang/UGP&Date)

# Paper

If you find this project useful, please consider citing:
```bibtex
    @inproceedings{zeng2025unlocking,
        title={Unlocking Generalization Power in LiDAR Point Cloud Registration},
        author={Zeng, Zhenxuan and Wu, Qiao and Zhang, Xiyu and Wu, Lin Yuanbo and An, Pei and Yang, Jiaqi and Wang, Ji and Wang, Peng},
        booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
        pages={22244--22253},
        year={2025}
    }
```

## Acknowlegdements

This codebase borrows from most notably [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences), [PREDATOR](https://github.com/prs-eth/OverlapPredator), [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch), [RPMNet](https://github.com/yewzijian/RPMNet) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork). Many thanks to the authors for generously sharing their codes!
