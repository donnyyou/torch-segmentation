# MT-Segmentation
```
@misc{mt-segmentation,
    author = {Ansheng You, Zhenhua Chai},
    title = {MT-Segmentation},
    howpublished = {\url{http://git.sankuai.com/users/youansheng/repos/mt-segmentation}},
    year = {2020}
}
```

This repository provides source code for most deep learning based cv problems. We'll do our best to keep this repository up-to-date.  If you do find a problem about this repository, please raise an issue or submit a pull request.


## Implemented Papers

- [Semantic Segmentation](http://git.sankuai.com/users/youansheng/repos/mt-segmentation/browse/model/nets)
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - PSPNet: Pyramid Scene Parsing Network
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    - Asymmetric Non-local Neural Networks for Semantic Segmentation
    
## QuickStart with TorchCV
Now only support Python3.x, pytorch 1.3.

```bash
pip3 install -r requirements.txt
cd lib/exts
sh make.sh
```


## Performances with MT-Segmentation
All the performances showed below fully reimplemented the papers' results.

#### Semantic Segmentation
- Cityscapes (Single Scale Whole Image Test): Base LR 0.01, Crop Size 769

| Model | Backbone | Train | Test | mIOU | BS | Iters | Scripts |
|:--------|:---------|:------|:------|:------|:------|:------|:------|
| [PSPNet]() | [3x3-Res101]() | train | val | 78.20 | 8 | 4W | [PSPNet]() |
| [DeepLabV3]() | [3x3-Res101]() | train | val | 79.13 | 8 | 4W | [DeepLabV3]() |

- ADE20K (Single Scale Whole Image Test): Base LR 0.02, Crop Size 520

| Model | Backbone | Train | Test | mIOU | PixelACC | BS | Iters | Scripts |
|:--------|:---------|:------|:------|:------|:------|:------|:------|:------|
| [PSPNet]() | [3x3-Res50]() | train | val | 41.52 | 80.09 | 16 | 15W | [PSPNet]() |
| [DeepLabv3]() | [3x3-Res50]() | train | val | 42.16 | 80.36 | 16 | 15W | [DeepLabV3]() |
| [PSPNet]() | [3x3-Res101]() | train | val | 43.60 | 81.30 | 16 | 15W | [PSPNet]() |
| [DeepLabv3]() | [3x3-Res101]() | train | val | 44.13 | 81.42 | 16 | 15W | [DeepLabV3]() |


## Commands with MT-Segmentation

Take PSPNet as an example. ("tag" could be any string, include an empty one.)
- Training

```bash
cd scripts/cityscapes/
bash run_fs_pspnet_cityscapes_seg.sh train tag
```

- Resume Training

```bash
cd scripts/cityscapes/
bash run_fs_pspnet_cityscapes_seg.sh train tag
```

- Validate

```bash
cd scripts/cityscapes/
bash run_fs_pspnet_cityscapes_seg.sh val tag
```

- Testing:

```bash
cd scripts/cityscapes/
bash run_fs_pspnet_cityscapes_seg.sh test tag
```

