# Cross-modal State Space Model for Real-time RGB-Thermal Wild Scene Semantic Segmentation
## Introduction
This repository contains the code for the paper "Cross-modal State Space Model for Real-time RGB-Thermal Wild Scene Semantic Segmentation," submitted to IROS 2025.
## Method 
![picture1](./figure/fig2.png)
The CM-SSM consists of two image encoders to extract the features of RGB and thermal images, four CM-SSA moudules to perform RGB-T feature fusion in four stages, and an MLP decoder to predict the semantic segmentation maps.
![picture2](./figure/fig3.png)
The CM-SS2D consists of three steps: 1) cross-modal selective scanning, 2) cross-modal state space modeling and 3) scan merging.
## Reqiurements
Python==3.9
Pytorch==2.0.1 
Cuda==11.8
mamba-ssm==1.0.1
selective-scan==0.0.1
mmcv==2.2.0
## Dataset and Results
| Models | Dataset  | mIoU | Weights|
|------|------------|------|--------------|
| CM-SSM    | CART       | 75.4   | [pth]()     |
| CM-SSM   | PST900       | 84.8    | [pth]()     |
