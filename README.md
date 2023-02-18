# HOI4D Action segmentation challenge
Please check out the HOI4D Challenge on the latest project website www.hoi4d.top !
## Overview
This code base provides a benchmark for the HOI4D challenge action segmentation task, and we provide the training code for two models, P4Transformer and PPTr.
## Challege
For this challege, you need submmit a pred.npy file(your predicted results) to the leaderboard. The file pred.npy is a ndarray:(892, 150) which is the prediction of test_wolabel.h5. You can download the example here: [Link](https://1drv.ms/u/s!ApQF_e_bw-USgjQCKg9hGJIijeqs?e=eGfohd)
## Install
These packages are needed:
```
torch
numpy
torchvision
```
This code is also based on the environment of pointnet++, so you should install it using following command:
```
cd ./modules
pip install .
```
## Usage
You can reproduce the result of PPTr or P4Transformer using:
```
python train_pptr.py --output-dir ./output
python train_p4.py --output-dir ./output
```
## Citation
```
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Yunze and Liu, Yun and Jiang, Che and Lyu, Kangbo and Wan, Weikang and Shen, Hao and Liang, Boqiang and Fu, Zhoujie and Wang, He and Yi, Li},
    title     = {HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21013-21022}
}
```
```
@inproceedings{wen2022point,
  title={Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding},
  author={Wen, Hao and Liu, Yunze and Huang, Jingwei and Duan, Bo and Yi, Li},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIX},
  pages={19--35},
  year={2022},
  organization={Springer}
}
```
```
@inproceedings{fan2021point,
  title={Point 4d transformer networks for spatio-temporal modeling in point cloud videos},
  author={Fan, Hehe and Yang, Yi and Kankanhalli, Mohan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14204--14213},
  year={2021}
}
```
