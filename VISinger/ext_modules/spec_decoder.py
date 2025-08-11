# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/12/8
# Email: skzheng@163.com
# Description: 解码成声谱图
import torch
from torch import nn

from VISinger import modules
from VISinger import commons


class SpecDecoder(nn.Module):
    '''
        解码器,将隐藏特征转换为声谱图,网络架构如下：
        类似WaveNet的扩张卷积（加残差连接）
        输入：隐藏编码,mask
        输出：声谱图编码
    '''
    def __init__(self,
            in_channels,
            out_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        spec = self.proj(x) * x_mask
        return spec
