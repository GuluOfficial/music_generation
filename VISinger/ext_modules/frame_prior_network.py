# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/7/11
# Description: 帧预处理模块
# The frame prior network is composed of multiple layers of one-dimensional convolution

import torch
from torch import nn
from torch.nn import functional as F


class FramePriorNetwork(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 out_channels,
                 n_layers=3,
                 condition_channels=256,
                 ):
        super(FramePriorNetwork, self).__init__()

        self.conv_layers = nn.ModuleList(
            [nn.Conv1d(
                in_channels=input_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=1,
            ) for i in range(n_layers)
            ]
        )
        self.condition_proj = nn.Conv1d(condition_channels, input_channels, 1)

    def forward(self, x, condition, x_mask):
        if condition is not None:
            x = x + self.condition_proj(condition)

        x = x * x_mask
        for conv in self.conv_layers:
            x = conv(x)
            x = F.gelu(x)
            x = x * x_mask

        return x
