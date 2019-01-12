# coding:utf-8
# Donny You
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat

from models.tools.module_helper import ModuleHelper


class SelfAttentionModule(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels,
                 out_channels=None, kernel_size=1, dilation=1, padding=0, stride=1, scale=1, bn_type=None):
        super(SelfAttentionModule, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.kernel_size = self._pair(kernel_size)
        self.dilation = self._pair(dilation)
        self.padding = self._pair(padding)
        self.stride = self._pair(stride)
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        # self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    @staticmethod
    def _pair(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, 2))

    def _out_size(self, size):
        return [(s + 2 * self.padding[i] -
                 self.dilation[i] * (self.kernel_size[i]-1) - 1) // self.stride[i] + 1 for i, s in enumerate(size)]

    def forward(self, x):
        b, _, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x)
        _, value_c, value_h, value_w = value.size()
        unfold_value = F.unfold(value, kernel_size=self.kernel_size,
                                dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_value_h, unfold_value_w = self._out_size([value_h, value_w])
        unfold_value = unfold_value.view(b, value_c, -1, unfold_value_h, unfold_value_w).contiguous()
        # print('unfold_value: {}'.format(unfold_value.size()))

        key = self.f_key(x)
        _, key_c, key_h, key_w = key.size()
        unfold_key = F.unfold(key, kernel_size=self.kernel_size,
                              dilation=self.dilation, padding=self.padding, stride=self.stride)
        unfold_key_h, unfold_key_w = self._out_size([key_h, key_w])
        unfold_key = unfold_key.view(b, key_c, -1, unfold_key_h, unfold_key_w).contiguous()
        # print('unfold_key: {}'.format(unfold_key.size()))

        assert unfold_value_h == unfold_key_h and unfold_value_w == unfold_key_w

        query = self.f_query(x)
        start_index = [self.dilation[i] * (self.kernel_size[i] // 2) - self.padding[i] for i in range(2)]
        query = query[:, :, start_index[0]::self.stride[0], start_index[1]::self.stride[1]].contiguous()
        query = query[:, :, :unfold_key_h, :unfold_key_w].contiguous()
        query = query.unsqueeze(2)
        # print('query: {}'.format(query.size()))

        sim_map = (unfold_key * query).sum(1, keepdim=True)
        sim_map = F.softmax(sim_map, 2)
        # print('sim_map: {}'.format(sim_map.size()))

        context = (sim_map * unfold_value).sum(2).contiguous()
        context = self.W(context)
        context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        # print('context: {}'.format(context.size()))
        return context


class SelfAttentionModuleV2(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels,
                 out_channels=None, kernel_size=1, dilation_list=None,
                 padding_list=None, stride=1, scale=1, bn_type=None):
        super(SelfAttentionModuleV2, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.kernel_size = self._pair(kernel_size)
        self.padding_list = padding_list
        self.dilation_list = dilation_list
        self.stride = self._pair(stride)
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        # self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    @staticmethod
    def _pair(x):
        if isinstance(x, (list, tuple)):
            return x
        return tuple(repeat(x, 2))

    def forward(self, x):
        b, _, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x)
        _, value_c, value_h, value_w = value.size()
        unfold_value_list = []
        for dilation, padding in zip(self.dilation_list, self.padding_list):
            unfold_value = F.unfold(value, kernel_size=self.kernel_size,
                                    dilation=dilation, padding=padding, stride=self.stride)
            unfold_value = unfold_value.view(b, value_c, -1, value_h, value_w).contiguous()
            # print('unfold_value: {}'.format(unfold_value.size()))
            unfold_value_list.append(unfold_value)

        unfold_value = torch.cat(unfold_value_list, 2)

        key = self.f_key(x)
        _, key_c, key_h, key_w = key.size()
        unfold_key_list = []
        for dilation, padding in zip(self.dilation_list, self.padding_list):
            unfold_key = F.unfold(key, kernel_size=self.kernel_size,
                                  dilation=dilation, padding=padding, stride=self.stride)
            unfold_key = unfold_key.view(b, key_c, -1, key_h, key_w).contiguous()
            # print('unfold_key: {}'.format(unfold_key.size()))
            unfold_key_list.append(unfold_key)

        unfold_key = torch.cat(unfold_key_list, 2)

        assert unfold_value.size(-2) == unfold_key.size(-2) and unfold_value.size(-1) == unfold_key.size(-1)

        query = self.f_query(x)
        query = query.unsqueeze(2)
        # print('query: {}'.format(query.size()))

        sim_map = (unfold_key * query).sum(1, keepdim=True)
        sim_map = F.softmax(sim_map, 2)
        # print('sim_map: {}'.format(sim_map.size()))

        context = (sim_map * unfold_value).sum(2).contiguous()
        context = self.W(context)
        context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        # print('context: {}'.format(context.size()))
        return context


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    in_x = torch.randn((2, 10, 256, 512)).cuda()
    self_attention = SelfAttentionModule(in_channels=10, key_channels=2, value_channels=5,
                                         kernel_size=5, dilation=6, padding=12)
    self_attention.cuda()
    params = self_attention.state_dict()
    import time
    for i in range(100):
        start_time = time.time()
        out_x = self_attention(in_x)
        torch.cuda.synchronize()
        print(time.time() - start_time)

    print(out_x.size())
