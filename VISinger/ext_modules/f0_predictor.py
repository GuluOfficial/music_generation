# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/7/11
# Description: F0预测模块
# F0 predictor consists of multiple FFT blocks

import torch
from torch import nn

from VISinger import attentions
from VISinger import commons


class F0Predictor(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 ):
        super(F0Predictor, self).__init__()

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.output_layer = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, x, x_lengths, x_mask):
        f0_emb = self.encoder(x * x_mask, x_mask)
        log_f0 = self.output_layer(f0_emb * x_mask)
        log_f0 = log_f0 * x_mask
        # [B, 1, T] -> [B, T]
        log_f0 = torch.squeeze(log_f0, dim=1)
        return log_f0

