# PyTorchCV-SemSeg: Open Source for Semantic Segmentation.
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/donnyyou/PyTorchCV-SemSeg}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based semantic segmentation. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.


## Implemented Papers

- [Semantic Segmentation](https://github.com/donnyyou/PyTorchCV-SemSeg/tree/master/methods)
    - PSPNet: Pyramid Scene Parsing Network
    - DeepLabV3: Rethinking Atrous Convolution for Semantic Image Segmentation
    - DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes

- CityScapes (Single Scale Whole Image Test): Base LR 0.01, Crop Size 769

| Checkpoints | Backbone | Train | Test | mIOU | BS | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet](https://drive.google.com/open?id=1bjQ8c-h1IBQPgp7DDwXl-U3tBo1lW6wB) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | 8 | 4W | [PSPNet](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_pspnet_cityscape_seg.sh) |
| [DeepLabV3](https://drive.google.com/open?id=15f--MUIMtiPHL8HyH_2A7EofJIPmA-oa) | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | 8 | 4W | [DeepLabV3](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/cityscape/run_fs_deeplabv3_cityscape_seg.sh) |


- ADE20K (Single Scale Whole Image Test): Base LR 0.02, Crop Size 520

| Checkpoints | Backbone | Train | Test | mIOU | PixelACC | BatchSize | Iters | Scripts |
|--------|:---------:|:------:|:------:|:------:|:------:|:------:|:------:|:------|
| [PSPNet]() | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [PSPNet](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_res50_pspnet_ade20k_seg.sh) |
| [DeepLabv3]() | [3x3-Res50](https://drive.google.com/open?id=1zPQLFd9c1yHfkQn5CWBCcEKmjEEqxsWx) | train | val | - | - | 16 | 15W | [DeepLabV3](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_res50_deeplabv3_ade20k_seg.sh) |
| [PSPNet]() | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [PSPNet](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_res101_pspnet_ade20k_seg.sh) |
| [DeepLabv3]() | [3x3-Res101](https://drive.google.com/open?id=1bUzCKazlh8ElGVYWlABBAb0b0uIqFgtR) | train | val | - | - | 16 | 15W | [DeepLabV3](https://github.com/donnyyou/PyTorchCV-SemSeg/blob/master/scripts/ade20k/run_fs_res101_deeplabv3_ade20k_seg.sh) |
