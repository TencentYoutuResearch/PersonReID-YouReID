# ReID torch 版本

## 功能
- 单/多GPU训练
- 分布式训练(DistributedDataParallel)
- 训练加测试
- fp16训练

## 训练
cd train && ./cmd.sh

## fp16性能
以baseline为例， 显卡试16GB的Titan V

| 模型  |Market1501 <br> mAP/rank-1|耗时| 注
|---|---|---|---|
|dist/baseline_b32_2gpu|88.44/95.69|22 M 13 s | 2gpu b32
|dist/baseline_b32_2gpu|88.46/95.40|53 M 57 s | 1gpu b32
|baseline|87.01/95.07|1 H 23 M 58 s|1gpu b128
|baseline_fp16_b|87.27/94.39|1 H 3 M 23 s|1gpu b64
|baseline_fp16|86.94/94.69|0 H 57 M 54 s|1gpu b128
|baseline_2gpu|87.74 / 94.92|1 H 1 M 2 s|2gpu b128
|baseline_fp16_2gpu|87.53/94.98|0 H 49 M 20 s|2gpu b128
|baseline_resnest|88.35/94.80|1 H 2 M 6 s|2gpu b128
|baseline_101_2gpu|88.97/95.31|1 H 17 M 46 s|2gpu b128
|baseline_101_fp16_2gpu|88.77/95.10|1 H 2 M 1 s|2gpu b128
|baseline_101_32x8d_weak_b64|90.52/95.78|3 H 17 M 17 s|2gpu 64
|baseline_101_32x8d_weak_fp16_b64|90.48/96.23|2 H 24 M 54 s|2gpu b64

## 示例

| 模型  |mAP/rank-1| backbone
|---|---|---|
|baseline|87.74/94.92|resnet50
||88.97/95.31|resnet101
||89.79/95.67|resnet101_32x8d
||90.52/95.78|resnet101_32x8d_weak
||90.46/95.96|resnet_101_ibn
|pcb|85.47/94.54|resnet50
|mgn|89.16/95.87|resnet50

