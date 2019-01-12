#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Donny You(youansheng@gmail.com)
# Some methods used by main methods.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from torch.optim import SGD, Adam, lr_scheduler

from utils.tools.logger import Logger as Log


class OptimScheduler(object):
    def __init__(self, configer):
        self.configer = configer

    def init_optimizer(self, net_params):
        optimizer = None
        if self.configer.get('optim', 'optim_method') == 'sgd':
            optimizer = SGD(net_params,
                            lr=self.configer.get('lr', 'base_lr'),
                            momentum=self.configer.get('optim', 'sgd')['momentum'],
                            weight_decay=self.configer.get('optim', 'sgd')['weight_decay'],
                            nesterov=self.configer.get('optim', 'sgd')['nesterov'])

        elif self.configer.get('optim', 'optim_method') == 'adam':
            optimizer = Adam(net_params,
                             lr=self.configer.get('lr', 'base_lr'),
                             betas=self.configer.get('optim', 'adam')['betas'],
                             eps=self.configer.get('optim', 'adam')['eps'],
                             weight_decay=self.configer.get('optim', 'adam')['weight_decay'])

        else:
            Log.error('Optimizer {} is not valid.'.format(self.configer.get('optim', 'optim_method')))
            exit(1)

        policy = self.configer.get('lr', 'lr_policy')

        scheduler = None
        if policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer,
                                            self.configer.get('lr', 'step')['step_size'],
                                            gamma=self.configer.get('lr', 'step')['gamma'])

        elif policy == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer,
                                                 self.configer.get('lr', 'multistep')['stepvalue'],
                                                 gamma=self.configer.get('lr', 'multistep')['gamma'])

        elif policy == 'lambda_poly':
            lambda_poly = lambda iters: pow((1.0 - iters / self.configer.get('solver', 'max_iters')), 0.9)
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)

        elif policy == 'lambda_cosine':
            lambda_cosine = lambda iters: (math.cos(math.pi * iters / self.configer.get('solver', 'max_iters'))
                                           + 1.0) / 2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)

        elif policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=self.configer.get('lr', 'plateau')['mode'],
                                                       factor=self.configer.get('lr', 'plateau')['factor'],
                                                       patience=self.configer.get('lr', 'plateau')['patience'],
                                                       threshold=self.configer.get('lr', 'plateau')['threshold'],
                                                       threshold_mode=self.configer.get('lr', 'plateau')['thre_mode'],
                                                       cooldown=self.configer.get('lr', 'plateau')['cooldown'],
                                                       min_lr=self.configer.get('lr', 'plateau')['min_lr'],
                                                       eps=self.configer.get('lr', 'plateau')['eps'])

        else:
            Log.error('Policy:{} is not valid.'.format(policy))
            exit(1)

        return optimizer, scheduler

    def update_optimizer(self, net, optim_method, lr_policy):
        self.configer.update(('optim', 'optim_method'), optim_method)
        self.configer.update(('lr', 'lr_policy'), lr_policy)
        optimizer, scheduler = self.init_optimizer(net)
        return optimizer, scheduler




