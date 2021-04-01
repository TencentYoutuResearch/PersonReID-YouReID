<p align="center"><img src="./docs/YouReID_Logo.png" width="300" ></p>

## Introduction

YouReID is a light research framework that implements some state-of-the-art person re-identification algorithms for some reid tasks and provides some strong baseline models.

### Major features
- [x] Simple design style, easy to use and customize. You can get started in 5 minutes.
- [x] Mixed precision and DistributedDataParallel training are supported, achieving higher efficiency.  You can run over the baseline model in 25 minutes using two 16GB V100 on the Market-1501 dataset.
- [x] Some strong baseline methods, including baseline, PCB, MGN.  Specially the performance of baseline model arrives mAP=87.65% and rank-1=94.80% on the Market-1501 dataset.
- [x] State-of-the-art methods for some reid tasks are supported.

## Model Zoo
this project provides the following algorithms and scripts to run them. Please see the details in the link provided in the description column

<table>
    <tr>
        <th>Field</th><th>ABBRV</th><th>Algorithms</th><th>Description</th><th>Status</th>
    </tr>
    <tr>
	<td rowspan="2">SL</td><td>CACENET</td><td><a href="https://arxiv.org/abs/2009.05250">Devil's in the Details: Aligning Visual Clues for Conditional Embedding in Person Re-Identification</a></td><td><a href="docs/CACENET/CACENET.md">CACENET.md</a></td><td>finished</td>
    </tr>
    <tr>
        <td>Pyramid</td><td><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.pdf">Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training</a></td><td><a href="docs/Pyramid/CVPR-2019-Pyramid.md">CVPR-2019-Pyramid.md</a></td><td>finished</td>
    </tr>
	<tr>
	<td>UDA</td><td>ACT</td><td><a href="https://arxiv.org/abs/1911.12512">Asymmetric Co-Teaching for Unsupervised Cross Domain Person Re-Identification</a></td><td><a href="docs/ACT/AAAI-2020-ACT.md">AAAI-2020-ACT.md</a></td><td>comming soon</td>
	</tr>
	<tr>
	<td>Occluded </td><td>PartNet</td><td><a href="https://arxiv.org/abs/1911.12512">Human Pose Information Discretization for Occluded Person Re-Identification</a></td><td><a href="docs/PartNet/PartNet.md">PartNet.md</a></td><td>finished</td>
	</tr>
	<tr>
	<td>Video </td><td>TSF</td><td><a href="https://arxiv.org/abs/1911.12512">Rethinking Temporal Fusion for Video-based Person Re-identification on Semantic and Time Aspect</a></td><td><a href="docs/TSF/AAAI-2020-TSF.md">AAAI-2020-TSF.md</a></td><td>comming soon</td>
	</tr>
	<tr>
	<td>Text </td><td>NAFS</td><td><a href="https://arxiv.org/pdf/2101.03036">Contextual Non-Local Alignment over Full-Scale Representation for Text-Based Person Search</a></td><td><a href="docs/NAFS/NAFS.md">NAFS.md</a></td><td>comming soon</td>
	</tr>
	<tr>
	<td>3D </td><td>Person-ReID-3D</td><td><a href="https://arxiv.org/pdf/2101.03036">Learning 3D Shape Feature for Texture-insensitive Person Re-identification</a></td><td><a href="docs/Person-ReID-3D/CVPR-2021-PR3D.md">CVPR-2021-PR3D.md</a></td><td>comming soon</td>
	</tr>
</table>

You also can find these models in [model_zoo](docs/model_zoo.md)
Specially we contribute some reid samples to opencv community, you can use these model in [opencv](https://github.com/opencv/opencv/pull/19108), and you also can visit them at [ReID_extra_testdata](https://github.com/ReID-Team/ReID_extra_testdata).
## Requirements and Preparation
Please install `Python>=3.6` and `PyTorch>=1.6.0`. 

#### Prepare Datasets
Download the public datasets(like market1501 and DukeMTMC), organize these datasets using the following format:

File Directory:
```
├── partitions.pkl
├── images
│ ├── 0000000_0000_000000.png
│ ├── 0000001_0000_000001.png
│ ├── ...
```

1. Rename the images in following convention:
"000000_000_000000.png" where the first substring splitted by underline is the person identity;
for the second substring, the first digit is the camera id and the rest is track id;
and the third substring is an image offset.

2. "partitions.pkl" file
This file contains a python dictionary storing meta data of the datasets, which contains folling key value pairs
"train_im_names": [list of image names] #storing a list of names of training images
"train_ids2labels":{"identity":label} #a map that maps the person identity string to a integer label
"val_im_names": [list of image names] #storing a list of names of validation images
"test_im_names": [list of image names] #storing a list of names of testing images
"test_marks"/"val_marks": [list of 0/1] #0/1 indicates if an image is in gallery

you can run tools/transform_format.py to get the formatted dataset or download from [formatted market1501](https://drive.google.com/file/d/1tqRV9ECq3zufuGzXpCvk3SF5jJEa51EB/view?usp=sharing)

## Geting Started

#### Clone this github repository:
```
git clone this repository
```

#### train
1. Configure basic settings in core/config
2. Define the network in net and register in the factory.py
3. Set the corresponding hyperparameters in the example yaml
4. set example.yaml path in config.yaml  
5. set port and gpu config in cmd.sh
5. cd train && ./cmd.sh

#### Quickly Started

```bash
cd train && ./cmd.sh
```

## Citation
If you are interested in our works, please cite our papers





