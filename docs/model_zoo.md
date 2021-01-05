# you-reid model zoo 

## Introduction

This file documents collection of models trained with you-reid framework. All numbers were obtained with 2 NVIDIA V100 GPU. The software in use were PyTorch 1.6, CUDA 10.1.

## Models

||Market1501<br>mAP&rank-1</br>|DukeMTMC<br>mAP&rank-1</br>|MSMT17<br>mAP&rank-1</br>|config|download|
|:-:|:-:|:-:|:-:|:-:|
|baseline|87.65/94.80|77.21/88.33|-|[config](../example/baseline/baseline_dist_bn.yaml)|[weight]() [log]()|


#### multi sources
we contribute some reid samples to opencv community, you can use these model in [opencv](https://github.com/opencv/opencv/pull/19108), and you also can visit them at [ReID_extra_testdata](https://github.com/ReID-Team/ReID_extra_testdata)
The following table shows the performance of these model

||Market1501<br>mAP&rank-1</br>|DukeMTMC<br>mAP&rank-1</br>|MSMT17<br>mAP&rank-1</br>|config|download|
|:-:|:-:|:-:|:-:|:-:|
|youtu_reid_baseline_lite|87.86/95.01|79.75/89.05|58.82/80.81|[config](../example/baseline/baseline_lite_multidataset.yaml)|[weight]() [log]()|
|youtu_reid_baseline_medium|90.75/96.32|83.38/91.56|65.30/85.08|[config](../example/baseline/baseline_medium_multidataset.yaml)|[weight]() [log]()|
|youtu_reid_baseline_large|91.85/96.73|84.40/91.88|68.68/87.04|[config](../example/baseline/baseline_large_multidataset.yaml)|[weight]() [log]()|
