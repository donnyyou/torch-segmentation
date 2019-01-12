#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)


import torch
from collections import OrderedDict

match_dict = {
    'conv1.0.weight': 'conv1.weight',
    'conv1.1.weight': 'bn1.weight',
    'conv1.1.bias': 'bn1.bias',
    'conv1.1.running_mean': 'bn1.running_mean',
    'conv1.1.running_var': 'bn1.running_var',
    'conv1.3.weight': 'conv2.weight',
    'conv1.4.weight': 'bn2.weight',
    'conv1.4.bias': 'bn2.bias',
    'conv1.4.running_mean': 'bn2.running_mean',
    'conv1.4.running_var': 'bn2.running_var',
    'conv1.6.weight': 'conv3.weight',
    'bn1.weight': 'bn3.weight',
    'bn1.bias': 'bn3.bias',
    'bn1.running_mean': 'bn3.running_mean',
    'bn1.running_var': 'bn3.running_var'
}


if __name__ == "__main__":
    ori_weight_file = '../../pretrained_model/resnet101-2a57e44d.pth'
    new_weight_file = '../../pretrained_model/pytorch-3x3resnet101-imagenet.pth'
    model_dict = torch.load(ori_weight_file)
    new_dict = OrderedDict()
    for key, value in model_dict.items():
        if key in match_dict.keys():
            new_dict[match_dict[key]] = value
        else:
            new_dict[key] = value

    torch.save(new_dict, new_weight_file)
