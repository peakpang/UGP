# Unlocking Generalization Power in LiDAR Point Cloud Registration (CVPR 2025)

In real-world environments, a LiDAR point cloud registration method with robust generalization capabilities (across varying distances and datasets) is crucial for ensuring safety in autonomous driving and other LiDAR-based applications. However, current methods fall short in achieving this level of generalization. To address these limitations, we propose UGP, a pruned framework designed to enhance generalization power for LiDAR point cloud registration. The core insight in UGP is the elimination of cross-attention mechanisms to improve generalization, allowing the network to concentrate on intra-frame feature extraction. Additionally, we introduce a progressive self-attention module to reduce ambiguity in large-scale scenes and integrate Bird’s Eye View (BEV) features to incorporate semantic information about scene elements. Together, these enhancements significantly boost the network’s generalization performance. We validated our approach through various generalization experiments in multiple outdoor scenes. In cross-distance generalization experiments on KITTI and nuScenes, UGP achieved state-of-the-art mean Registration Recall rates of 94.5\% and 91.4\%, respectively. In cross-dataset generalization from nuScenes to KITTI, UGP achieved a state-of-the-art mean Registration Recall of 90.9\%.

## News

20250227 - Our paper has been accepted by CVPR'25!

## Overview of UGP

<div align="center">
<img src=assets\arch.png>
</div>

## Acknowlegdements

We thank [FCGF](https://github.com/chrischoy/FCGF) for the wonderful baseline, [SC2-PCR](https://github.com/ZhiChen902/SC2-PCR) for a powerful and fast alternative registration algorithm.

We would also like to thank [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) and [waymo-open-dataset](https://github.com/waymo-research/waymo-open-dataset) for the convenient dataset conversion codes.
