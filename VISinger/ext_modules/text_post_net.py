# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/7/11
# Description: TextEncoder后处理网络

import torch
from torch import nn

from VISinger.ext_modules.frame_prior_network import FramePriorNetwork
from VISinger.ext_modules.f0_predictor import F0Predictor
from VISinger.modules import LayerNorm
from VISinger import commons
from VISinger.utils.pitch_utils import f0_to_coarse
from VISinger.ext_modules.mel_predict_network import MelPredictNetwork


class TextPostNet(nn.Module):
    def __init__(self,
                 hidden_channels,
                 out_channels,
                 filter_channels,
                 n_heads,
                 f0_predictor_layers,
                 kernel_size,
                 p_dropout,
                 max_position_embeddings=2000,
                 n_layers_frame_prior=3,
                 ):
        super(TextPostNet, self).__init__()

        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_channels)
        # self.layer_norm = LayerNorm(hidden_channels)

        self.f0_predictor = F0Predictor(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=f0_predictor_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        self.mel_predict_net = MelPredictNetwork(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            condition_channels=hidden_channels,
            n_heads=n_heads,
            n_layers=4,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            out_channels=80
        )
        self.frame_prior_net = FramePriorNetwork(
            input_channels=80,
            out_channels=hidden_channels,
            n_layers=n_layers_frame_prior,
            condition_channels=hidden_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(300, hidden_channels)
        self.out_channels = out_channels

    def forward(self, x, x_lengths, f0_score=None, x_mask=None, position_ids=None, past_key_values_length: int = 0):
        input_shape = x.size()
        seq_length = input_shape[-1]
        embeddings = torch.transpose(x, 1, -1)
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]
            position_embeddings = self.position_embeddings(position_ids)

            embeddings += position_embeddings
            embeddings = torch.transpose(embeddings, 1, -1)
            embeddings = embeddings
            # embeddings = self.layer_norm(embeddings)

        x_emb = embeddings * x_mask
        # 基频预测
        # 分布生成处理
        log_f0 = self.f0_predictor(x_emb, x_lengths, x_mask)
        if f0_score is None:
            f0_score = f0_to_coarse(torch.exp(log_f0) - 1.0)
        f0_emb = self.f0_emb(f0_score)
        f0_emb = torch.transpose(f0_emb, 1, -1)
        f0_emb = f0_emb * x_mask

        mel_predict = self.mel_predict_net(x_emb, f0_emb, x_mask)

        x_hidden = self.frame_prior_net(mel_predict, f0_emb, x_mask)
        # 维度映射，生成均值、方差
        stats = self.proj(x_hidden * x_mask) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)

        return m, logs, log_f0, mel_predict
