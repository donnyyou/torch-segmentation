#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Class for the Semantic Segmentation Data Loader.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from torch.utils import data

import datasets.tools.cv2_aug_transforms as cv2_aug_trans
import datasets.tools.pil_aug_transforms as pil_aug_trans
import datasets.tools.transforms as trans
from datasets.fs_data_loader import FSDataLoader
from datasets.rs_data_loader import RSDataLoader
from datasets.tools.collate import collate
from utils.tools.logger import Logger as Log


class SegDataLoader(object):

    def __init__(self, configer):
        self.configer = configer

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_train_transform = pil_aug_trans.PILAugCompose(self.configer, split='train')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_train_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='train')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        if self.configer.get('data', 'image_tool') == 'pil':
            self.aug_val_transform = pil_aug_trans.PILAugCompose(self.configer, split='val')
        elif self.configer.get('data', 'image_tool') == 'cv2':
            self.aug_val_transform = cv2_aug_trans.CV2AugCompose(self.configer, split='val')
        else:
            Log.error('Not support {} image tool.'.format(self.configer.get('data', 'image_tool')))
            exit(1)

        self.img_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(div_value=self.configer.get('normalize', 'div_value'),
                            mean=self.configer.get('normalize', 'mean'),
                            std=self.configer.get('normalize', 'std')), ])

        self.label_transform = trans.Compose([
            trans.ToLabel(),
            trans.ReLabel(255, -1), ])

    def get_trainloader(self):
        if self.configer.exists('train', 'loader') and self.configer.get('train', 'loader') == 'rs':
            trainloader = data.DataLoader(
                RSDataLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             label_transform=self.label_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

        else:
            trainloader = data.DataLoader(
                FSDataLoader(root_dir=self.configer.get('data', 'data_dir'), dataset='train',
                             aug_transform=self.aug_train_transform,
                             img_transform=self.img_transform,
                             label_transform=self.label_transform,
                             configer=self.configer),
                batch_size=self.configer.get('train', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'),
                shuffle=True, drop_last=self.configer.get('data', 'drop_last'),
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('train', 'data_transformer')
                )
            )

            return trainloader

    def get_valloader(self, dataset=None):
        dataset = 'val' if dataset is None else dataset
        if self.configer.get('method') == 'fcn_segmentor':
            valloader = data.DataLoader(
                FSDataLoader(root_dir=self.configer.get('data', 'data_dir'), dataset=dataset,
                             aug_transform=self.aug_val_transform,
                             img_transform=self.img_transform,
                             label_transform=self.label_transform,
                             configer=self.configer),
                batch_size=self.configer.get('val', 'batch_size'), pin_memory=True,
                num_workers=self.configer.get('data', 'workers'), shuffle=False,
                collate_fn=lambda *args: collate(
                    *args, trans_dict=self.configer.get('val', 'data_transformer')
                )
            )

            return valloader

        else:
            Log.error('Method: {} loader is invalid.'.format(self.configer.get('method')))
            return None


if __name__ == "__main__":
    # Test data loader.
    pass
