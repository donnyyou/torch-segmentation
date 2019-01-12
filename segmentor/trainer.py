#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You (youansheng@gmail.com)
# Class Definition for Semantic Segmentation.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import cv2
import numpy as np
import torch
import torch.nn as nn

from datasets.seg_data_loader import SegDataLoader
from loss.loss_manager import LossManager
from segmentor.tools.module_runner import ModuleRunner
from segmentor.tools.optim_scheduler import OptimScheduler
from models.model_manager import ModelManager
from utils.tools.average_meter import AverageMeter
from utils.tools.logger import Logger as Log
from val.scripts.seg_running_score import SegRunningScore
from vis.seg_visualizer import SegVisualizer


#


class Trainer(object):
    """
      The class for Pose Estimation. Include train, val, val & predict.
    """
    def __init__(self, configer):
        self.configer = configer
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.seg_running_score = SegRunningScore(configer)
        self.seg_visualizer = SegVisualizer(configer)
        self.seg_loss_manager = LossManager(configer)
        self.module_runner = ModuleRunner(configer)
        self.seg_model_manager = ModelManager(configer)
        self.seg_data_loader = SegDataLoader(configer)
        self.optim_scheduler = OptimScheduler(configer)

        self.seg_net = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None

        self._init_model()

    def _init_model(self):
        self.seg_net = self.seg_model_manager.semantic_segmentor()
        self.seg_net = self.module_runner.load_net(self.seg_net)

        Log.info('Params Group Method: {}'.format(self.configer.get('optim', 'group_method')))
        if self.configer.get('optim', 'group_method') == 'decay':
            params_group = self.group_weight(self.seg_net)
        else:
            assert self.configer.get('optim', 'group_method') is None
            params_group = self._get_parameters()

        self.optimizer, self.scheduler = self.optim_scheduler.init_optimizer(params_group)

        self.train_loader = self.seg_data_loader.get_trainloader()
        self.val_loader = self.seg_data_loader.get_valloader()

        self.pixel_loss = self.seg_loss_manager.get_seg_loss()

    @staticmethod
    def group_weight(module):
        group_decay = []
        group_no_decay = []
        for m in module.modules():
            if isinstance(m, nn.Linear):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            elif isinstance(m, nn.modules.conv._ConvNd):
                group_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if hasattr(m, 'weight'):
                    group_no_decay.append(m.weight)
                if hasattr(m, 'bias'):
                    group_no_decay.append(m.bias)

        assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups

    def _get_parameters(self):
        bb_lr = []
        nbb_lr = []
        params_dict = dict(self.seg_net.named_parameters())
        for key, value in params_dict.items():
            if 'backbone' not in key:
                nbb_lr.append(value)
            else:
                bb_lr.append(value)

        params = [{'params': bb_lr, 'lr': self.configer.get('lr', 'base_lr')},
                  {'params': nbb_lr, 'lr': self.configer.get('lr', 'base_lr') * self.configer.get('lr', 'nbb_mult')}]
        return params

    def __train(self):
        """
          Train function of every epoch during train phase.
        """
        self.seg_net.train()
        start_time = time.time()

        for i, data_dict in enumerate(self.train_loader):
            if self.configer.get('lr', 'metric') == 'iters':
                self.scheduler.step(self.configer.get('iters'))
            else:
                self.scheduler.step(self.configer.get('epoch'))

            if self.configer.get('lr', 'is_warm'):
                self.module_runner.warm_lr(self.configer.get('iters'),
                                           self.scheduler, self.optimizer, backbone_list=[0,])
            inputs = data_dict['img']
            targets = data_dict['labelmap']
            self.data_time.update(time.time() - start_time)
            # Change the data type.
            # inputs, targets = self.module_runner.to_device(inputs, targets)

            # Forward pass.
            outputs = self.seg_net(inputs)
            # outputs = self.module_utilizer.gather(outputs)
            # Compute the loss of the train batch & backward.
            loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
            if self.configer.exists('train', 'loader') and self.configer.get('train', 'loader') == 'rs':
                batch_size = self.configer.get('train', 'batch_size')*self.configer.get('train', 'batch_per_gpu')
                self.train_losses.update(loss.item(), batch_size)
            else:
                self.train_losses.update(loss.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update the vars of the train phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()
            self.configer.plus_one('iters')

            # Print the log info & reset the states.
            if self.configer.get('iters') % self.configer.get('solver', 'display_iter') == 0:
                Log.info('Train Epoch: {0}\tTrain Iteration: {1}\t'
                         'Time {batch_time.sum:.3f}s / {2}iters, ({batch_time.avg:.3f})\t'
                         'Data load {data_time.sum:.3f}s / {2}iters, ({data_time.avg:3f})\n'
                         'Learning rate = {3}\tLoss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                         self.configer.get('epoch'), self.configer.get('iters'),
                         self.configer.get('solver', 'display_iter'),
                         self.module_runner.get_lr(self.optimizer), batch_time=self.batch_time,
                         data_time=self.data_time, loss=self.train_losses))
                self.batch_time.reset()
                self.data_time.reset()
                self.train_losses.reset()

            if self.configer.get('iters') == self.configer.get('solver', 'max_iters'):
                break

            # Check to val the current model.
            if self.configer.get('iters') % self.configer.get('solver', 'test_interval') == 0:
                self.__val()

        self.configer.plus_one('epoch')

    def __val(self, data_loader=None):
        """
          Validation function during the train phase.
        """
        self.seg_net.eval()
        start_time = time.time()

        data_loader = self.val_loader if data_loader is None else data_loader
        for j, data_dict in enumerate(data_loader):
            inputs = data_dict['img']
            targets = data_dict['labelmap']

            with torch.no_grad():
                # Change the data type.
                inputs, targets = self.module_runner.to_device(inputs, targets)
                # Forward pass.
                outputs = self.seg_net(inputs)
                # Compute the loss of the val batch.
                loss = self.pixel_loss(outputs, targets, gathered=self.configer.get('network', 'gathered'))
                outputs = self.module_runner.gather(outputs)

            self.val_losses.update(loss.item(), inputs.size(0))
            self._update_running_score(outputs[-1], data_dict['meta'])
            # self.seg_running_score.update(pred.max(1)[1].cpu().numpy(), targets.cpu().numpy())

            # Update the vars of the val phase.
            self.batch_time.update(time.time() - start_time)
            start_time = time.time()

        self.configer.update(['performance'], self.seg_running_score.get_mean_iou())
        self.configer.update(['val_loss'], self.val_losses.avg)
        self.module_runner.save_net(self.seg_net, save_mode='performance')
        self.module_runner.save_net(self.seg_net, save_mode='val_loss')

        # Print the log info & reset the states.
        Log.info(
            'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
            'Loss {loss.avg:.8f}\n'.format(
                batch_time=self.batch_time, loss=self.val_losses))
        Log.info('Mean IOU: {}\n'.format(self.seg_running_score.get_mean_iou()))
        Log.info('Pixel ACC: {}\n'.format(self.seg_running_score.get_pixel_acc()))
        self.batch_time.reset()
        self.val_losses.reset()
        self.seg_running_score.reset()
        self.seg_net.train()

    def _update_running_score(self, pred, metas):
        pred = pred.permute(0, 2, 3, 1)
        for i in range(pred.size(0)):
            ori_img_size = metas[i]['ori_img_size']
            border_size = metas[i]['border_size']
            ori_target = metas[i]['ori_target']
            total_logits = cv2.resize(pred[i, :border_size[1], :border_size[0]].cpu().numpy(),
                                      tuple(ori_img_size), interpolation=cv2.INTER_CUBIC)
            labelmap = np.argmax(total_logits, axis=-1)
            self.seg_running_score.update(labelmap[None], ori_target[None])

    def train(self):
        # cudnn.benchmark = True
        if self.configer.get('network', 'resume') is not None and self.configer.get('network', 'resume_val'):
            self.__val()

        while self.configer.get('iters') < self.configer.get('solver', 'max_iters'):
            self.__train()

        self.__val(data_loader=self.seg_data_loader.get_valloader(dataset='val'))
        self.__val(data_loader=self.seg_data_loader.get_valloader(dataset='train'))


if __name__ == "__main__":
    # Test class for pose estimator.
    pass
