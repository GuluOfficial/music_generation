# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/8/4
# Description: 帧预处理模块
# The frame prior network is composed of multiple layers of one-dimensional convolution

import torch
from torch import nn
from torch.nn import functional as F
from VISinger import attentions


class MelPredictNetwork(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 condition_channels,
                 filter_channels,
                 n_heads,
                 n_layers=2,
                 kernel_size=3,
                 p_dropout=0.1,
                 out_channels=80,
                 ):
        super(MelPredictNetwork, self).__init__()

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.condition_proj = nn.Conv1d(condition_channels, input_channels, 1)
        self.output_layer = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, condition, x_mask):
        if condition is not None:
            x = x + self.condition_proj(condition)

        x = x * x_mask
        x = self.encoder(x, x_mask) * x_mask
        x = self.output_layer(x) * x_mask

        return x
