# you-reid

you-reid is a research framework that implements some state-of-the art person re-identification algorithms


## Model Zoo
this project provides the following algorithms and scripts to run them. Please see the details in the link provided in the description column

<table>
    <tr>
        <th>Field</th><th>ABBRV</th><th>Algorithms</th><th>Description</th><th>Status</th>
    </tr>
    <tr>
	<td rowspan="3">SL</td><td>CACENET</td><td><a href="https://arxiv.org/abs/2009.05250">Devil's in the Details: Aligning Visual Clues for Conditional Embedding in Person Re-Identification</a></td><td><a href="docs/CACENET/CACENET.md">CACENET.md</a></td><td>finished</td>
    </tr>
    <tr>
        <td>Pyramid</td><td><a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Zheng_Pyramidal_Person_Re-IDentification_via_Multi-Loss_Dynamic_Training_CVPR_2019_paper.pdf">Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training</a></td><td><a href="docs/Pyramid/CVPR-2019-Pyramid.md">CVPR-2019-Pyramid.md</a></td><td>coming soon</td>
    </tr>
    <tr>
        <td>VAAL</td><td><a href="https://arxiv.org/abs/1912.01300">Viewpoint-Aware Loss with Angular Regularization for Person Re-Identification</a></td><td><a href="docs/VAAL/AAAI-2020-VAAL.md">AAAI-2020-VAAL.md</a></td><td>coming soon</td>
    </tr>
	<tr>
	<td>UDA</td><td>ACT</td><td><a href="https://arxiv.org/abs/1911.12512">Asymmetric Co-Teaching for Unsupervised Cross Domain Person Re-Identification</a></td><td><a href="docs/ACT/AAAI-2020-ACT.md">AAAI-2020-ACT.md</a></td><td>coming soon</td>
	</tr>
	<tr>
	<td>Occluded </td><td>PartNet</td><td><a href="https://arxiv.org/abs/1911.12512">Human Pose Information Discretization for Occluded Person Re-Identification</a></td><td><a href="docs/PartNet/PartNet.md">PartNet.md</a></td><td>coming soon</td>
	</tr>
	<tr>
	<td>Video </td><td>TSF</td><td><a href="https://arxiv.org/abs/1911.12512">Rethinking Temporal Fusion for Video-based Person Re-identification on Semantic and Time Aspect</a></td><td><a href="docs/TSF/AAAI-2020-TSF.md">AAAI-2020-TSF.md</a></td><td>coming soon</td>
	</tr>
	<tr>
	<td>Text </td><td>NAFS</td><td><a href="https://arxiv.org/abs/1911.12512">NAFS</a></td><td><a href="docs/NAFS/NAFS.md">NAFS.md</a></td><td>coming soon</td>
	</tr>
	<tr>
	<td>3D </td><td>Person-ReID-3D</td><td><a href="https://arxiv.org/abs/1911.12512">Person-ReID-3D</a></td><td><a href="docs/Person-ReID-3D/Person-ReID-3D.md">Person-ReID-3D.md</a></td><td>coming soon</td>
	</tr>
</table>

You also can find these models in [model_zoo](docs/model_zoo.md)
## Requirements and Preparation
Please install `Python>=3.6` and `PyTorch>=1.6.0`. 

#### Prepare Datasets
Download the public datasets(like market1501 and DukeMTMC), organize these datasets using the following format:

File Directory:
├── partitions.pkl
├── images
│ ├── 0000000_0000_000000.png
│ ├── 0000001_0000_000001.png
│ ├── ...

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

## Geting Started

#### Clone this github repository:
```
git clone this github repository
```

#### train
1. Configure basic settings in core/config
2. Define the network in net and register in the factory.py
3. Set the corresponding hyperparameters in the example yaml
4. set example.yaml path in config.yaml  
5. set port and gpu config in cmd.sh
5. cd train && ./cmd.sh

###### Quickly Started

```bash
cd train && ./cmd.sh
```

## Citation
If you are interested in our works, please cite our papers
```
@article{yu2020devil,
  title={Devil's in the Details: Aligning Visual Clues for Conditional Embedding in Person Re-Identification},
  author={Yu, Fufu and Jiang, Xinyang and Gong, Yifei and Zhao, Shizhen and Guo, Xiaowei and Zheng, Wei-Shi and Zheng, Feng and Sun, Xing},
  journal={arXiv e-prints},
  pages={arXiv--2009},
  year={2020}
}
@inproceedings{zheng2019pyramidal,
  title={Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training},
  author={Zheng, Feng and Deng, Cheng and Sun, Xing and Jiang, Xinyang and Guo, Xiaowei and Yu, Zongqiao and Huang, Feiyue and Ji, Rongrong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8514--8522},
  year={2019}
}
```




