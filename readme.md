# ReID torch 版本
## 功能
- 单/多GPU训练
- 训练加测试
- fp16训练

## fp16性能
以baseline为例， 显卡试16GB的Titan V

| 模型  |Market1501 <br> mAP/rank-1|耗时| 注
|---|---|---|---|
|baseline|87.01/95.07|1 H 23 M 58 s|1gpu b128
|baseline_fp16_b|87.27/94.39|1 H 3 M 23 s|1gpu b64
|baseline_fp16|86.94/94.69|0 H 57 M 54 s|1gpu b128
|baseline_2gpu|87.74 / 94.92|1 H 1 M 2 s|2gpu b128
|baseline_fp16_2gpu|87.53/94.98|0 H 49 M 20 s|2gpu b128
|baseline_101_2gpu|88.97/95.31|1 H 17 M 46 s|2gpu b128
|baseline_101_fp16_2gpu|88.77/95.10|1 H 2 M 1 s|2gpu b128
|baseline_101_32x8d_fp16|88.85/95.31|1 H 40 M 22 s|2gpu b128
|baseline_101_32x8d_weak_b64|90.52/95.78|3 H 17 M 17 s|2gpu 64
|baseline_101_32x8d_weak_fp16_b64|90.48/96.23|2 H 24 M 54 s|2gpu b64

## 示例

| 模型  |mAP/rank-1| backbone
|---|---|---|
|baseline|87.74/94.92|resnet50
||88.97/95.31|resnet101

