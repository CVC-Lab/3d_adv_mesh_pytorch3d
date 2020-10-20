
# Can 3D Adversarial Logos Clock Humans? #

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[Can 3D Adversarial Logos Clock Humans?]()

Tianlong Chen\*, Yi Wang\*, Jingyang Zhou*, Sijia Liu, Shiyu Chang, Chandrajit Bajaj, Zhangyang Wang

This repo is the updated version and reimplmentation using pytorch3d.




## Overview

Examples of our 3D adversarial logo attack on different3D object meshes to fool a YOLOV2 detector. 

![](./doc_imgs/intro.png)



## Methodology

![](./doc_imgs/methods.png)


## Preliminaries

You need to install pytorch3d to make most of the code.
For faster-rcnn detector, we use: https://github.com/potterhsu/easy-faster-rcnn.pytorch
Download and put it in the folder "faster_rcnn", including the checkpoints they provided for demo, which is the pretrained model.



## Citation

If you are use this code for you research, please cite our paper.

```
@article{chen2020can,
  title={Can 3D Adversarial Logos Cloak Humans?},
  author={Chen, Tianlong and Wang, Yi and Zhou, Jingyang and Liu, Sijia and Chang, Shiyu and Bajaj, Chandrajit and Wang, Zhangyang},
  journal={arXiv preprint arXiv:2006.14655},
  year={2020}
}
```

