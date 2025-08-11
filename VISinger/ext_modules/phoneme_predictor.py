# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/7/11
# Description: 音素预测模块
# The phoneme predictor consists of two layers of FFT

import torch
from torch import nn
from torch.nn import functional as F

from VISinger import attentions
from VISinger import commons


class PhonemePredictor(torch.nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_vocab,
                 n_layers=2,
                 kernel_size=3,
                 p_dropout=0.1,
                 ):
        super(PhonemePredictor, self).__init__()

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.output_layer = nn.Conv1d(hidden_channels, n_vocab, 1)

    def forward(self, x, x_mask):
        phoneme_emb = self.encoder(x * x_mask, x_mask)
        log_phoneme_prop = F.log_softmax(self.output_layer(phoneme_emb) * x_mask, dim=1)
        # [B, C, T] -> [B, T, C]
        log_phoneme_prop = torch.transpose(log_phoneme_prop, 1, 2)

        return log_phoneme_prop
