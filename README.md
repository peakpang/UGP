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

Coming soon.

## Acknowlegdements

This codebase borrows from most notably [GeoTransformer](https://github.com/qinzheng93/GeoTransformer), [CoFiNet](https://github.com/haoyu94/Coarse-to-fine-correspondences), [PREDATOR](https://github.com/prs-eth/OverlapPredator), [D3Feat](https://github.com/XuyangBai/D3Feat.pytorch), [RPMNet](https://github.com/yewzijian/RPMNet) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork). Many thanks to the authors for generously sharing their codes!
