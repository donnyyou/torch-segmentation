# PyTorchCV-SemSeg: Semantic Segmentation ported from PyTorchCV
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/youansheng/PyTorchCV-SemSeg}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based cv problems. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.


## Implemented Papers

- [Semantic Segmentation](https://github.com/youansheng/PyTorchCV-SemSeg/tree/master/methods)
    - OCNet: Object Context Network for Scene Parsing
    - PSPNet: Pyramid Scene Parsing Network
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
    


## Performances with PyTorchCV-SemSeg

- CityScapes (Single Scale Whole Image Test)

| Model | Backbone | Train | Test | mIOU | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1bjQ8c-h1IBQPgp7DDwXl-U3tBo1lW6wB) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 78.13 | 8 | 40000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_pspnet_cityscape_seg.sh) |
| [DeepLabV3](https://drive.google.com/open?id=1hyoXXAZW2-yu4XYHKF6c3FaD8BM6g6qZ) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 78.64 | 8 | 40000 | [DeepLabV3](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_deeplabv3_cityscape_seg.sh) |
| [Base-OCNet](https://drive.google.com/open?id=1n4yYrVq1lzT7Q0HhMgbB2TJcppAjmuT0) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 79.16 | 8 | 40000| [Base-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_baseocnet_cityscape_seg.sh) |
| [Pyramid-OCNet](https://drive.google.com/open?id=1oXiMpIxbcfoFC4xMZmhJ-c3yxpzRFcAS) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 78.87 | 8 | 40000| [Pyramid-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_pyramidocnet_cityscape_seg.sh) |
| [ASP-OCNet](https://drive.google.com/open?id=1_jPHJmqnej6tCK3CB2YSDK3xTU6AhYMW) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train-fine | val | 79.23 | 8 | 40000 | [ASP-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_aspocnet_cityscape_seg.sh) |


- ADE20K (Single Scale Whole Image Test)

| Model | Backbone | Train | Test | mIOU | PixelACC | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.65 | 79.64 | 16 | 75000 | [PSPNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_pspnet_ade20k_seg.sh) |
| [DeepLabv3](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.15 | 79.81 | 16 | 75000 | [DeepLabV3](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_deeplabv3_ade20k_seg.sh) |
| [Base-OCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 40.48 | 79.75 | 16 | 75000 | [Base-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_baseocnet_ade20k_seg.sh) |
| [Pyramid-OCNet](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.03 | 79.74 | 16 | 75000 | [Pyramid-OCNet](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_pyramidocnet_ade20k_seg.sh) |
| [ASP-OCNetV1](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.47 | 80.05 | 16 | 75000 | [ASP-OCNetV1](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_aspocnet_ade20k_seg.sh) |
| [ASP-OCNetV2](https://drive.google.com/open?id=1Q6oYBpq9Y53z_CJz7Km9BaiSVJjcHP4h) | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | 41.75 | 79.96 | 16 | 75000 | [ASP-OCNetV2](https://github.com/youansheng/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_resnet50_aspocnetv2_ade20k_seg.sh) |
