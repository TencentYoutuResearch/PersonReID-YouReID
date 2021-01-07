# Devil’s in the Details: Aligning Visual Clues for Conditional Embedding in Person Re-Identification

## Introduction
This is the tensorflow implementation of our paper "Devil’s in the Details: Aligning Visual Clues for Conditional Embedding in Person Re-Identification".
In this paper:
1) CACE-Net is able to integrate both visual clue alignment and conditional feature embedding into a unified ReID framework
2) Instead of using a pre-defined Adjacency Matrix, our CACE-Net uses a novel correspondence attention module where the visual clues is automatically predicted and dynamically adjusted during training

![image](pipeline.png)

## requirements
python >= 3.6
tensorflow >= 1.8

## train
### Prepare Datasets
1. File Directory:
├── partitions.pkl
├── images
│  ├── 0000000_0000_000000.png
│  ├── 0000001_0000_000001.png
│  ├── ...

2. Rename the images in following convention:
"000000_000_000000.png"
where the first substring splitted by underline is the person identity;
for the second substring, the first digit is the camera id and the rest is track id;
and the third substring is an image offset. 

3. "partitions.pkl" file
This file contains a python dictionary storing meta data of the datasets, which contains folling key value pairs
"train_im_names": [list of image names] #storing a list of names of training images
"train_ids2labels":{"identity":label} #a map that maps the person identity string to a integer label
"val_im_names": [list of image names] #storing a list of names of validation images
"test_im_names": [list of image names] #storing a list of names of testing images
"test_marks"/"val_marks": [list of 0/1] #0/1 indicates if an image is in gallery

### train 
1. Configure basic settings in core/config
2. Define the network in net and register in the factory.py
3. Set the corresponding hyperparameters in the experiment yaml
4. set experiment.yaml path in config.yaml 
5. python train.py

## test
1. Configure settings in eval_config.yaml， pay attetion to set train_mode false
2. register in val.save_pb
3. python evalution.py

## example
```yaml
yaml: 'experiment/graph/cacenet.yaml'
```
market: [mAP: 90.11%], [cmc1: 95.84%]

## peformance
![image](performance.png) 

## Citation

If you find this code useful, please cite the following paper:

```
@article{jiang2020devil,
  title={Devil's in the Detail: Graph-based Key-point Alignment and Embedding for Person Re-ID},
  author={Jiang, Xinyang and Yu, Fufu and Gong, Yifei and Zhao, Shizhen and Guo, Xiaowei and Huang, Feiyue and Zheng, Wei-Shi and Sun, Xing},
  journal={arXiv preprint arXiv:2009.05250},
  year={2020}
}
```




