import torch
import torch.nn as nn

from transformer_melody.durationpredictor import DurationPredictor, StochasticDurationPredictor
from transformer_melody.modules import Encoder



class Lyric2Duration(nn.Module):
    def __init__(self, 
                 input_dim,hidden_dim, enc_layers,enc_heads,enc_pf_dim,enc_dropout, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 use_sdp,
                 device):
        super().__init__()
        
        self.encoder = Encoder(input_dim,hidden_dim, enc_layers,enc_heads,enc_pf_dim,enc_dropout,device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.use_sdp = use_sdp
        gin_channels = 10
        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_dim, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_dim, 256, 3, 0.5, gin_channels=gin_channels)
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
  
    def infer_duration(self, src, noise_scale=1, length_scale=1, noise_scale_w=1.):
        
        src_mask = self.make_src_mask(src)
        encoder_outputs = self.encoder(src, src_mask)

        x = torch.transpose(encoder_outputs, 1, -1)
        x_mask = src_mask.squeeze(1)

        g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        # 此时是预测的时长序列
        w = torch.exp(logw) * x_mask * length_scale 
        return w

    def forward(self, src, durations):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        x = torch.transpose(enc_src, 1, -1)
        x_mask = src_mask.squeeze(1)

        if self.use_sdp:

            l_length = self.dp(x, x_mask, durations)
            l_length = l_length / torch.sum(x_mask)

        else:
      
            logw_ = torch.log(durations + 1e-6) * x_mask
            logw = self.dp(x, x_mask)
            l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 
        
        return l_length