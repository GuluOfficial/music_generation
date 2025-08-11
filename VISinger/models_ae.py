import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from VISinger import commons
from VISinger import modules
from VISinger import attentions
from VISinger import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from VISinger.commons import init_weights, get_padding

from VISinger.ext_modules import PhonemePredictor, TextPostNet as TextPostNet
from VISinger.ext_modules import SpecDecoder
from VISinger.utils import utils


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.spec_flows = nn.ModuleList()
        self.spec_flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.spec_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.spec_flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.spec_flows
            assert w is not None

            logdet_tot_q = 0 
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1) 
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
            return nll + logq # [b]
        else:
            flows = list(reversed(self.spec_flows))
            flows = flows[:-2] + [flows[-1]] # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)

        self.conv_3 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_3 = modules.LayerNorm(filter_channels)

        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        # # add by zsk
        x = self.conv_3(x * x_mask)
        x = torch.relu(x)
        x = self.norm_3(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)
        x = torch.sigmoid(x)
        x = x * x_mask
        duration_predict = torch.squeeze(x, dim=1)
        return duration_predict


class TextEncoder(nn.Module):
    """
        Transformer FFT Blocks 构成的Encoder,最后加一个PointWise Conv进行映射；
        输入：音素序列
        输出：h_text(Encoder 输出), mean, log(seita), PointWise Conv输出
    """
    def __init__(
            self,
            n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            n_vocab_pitch,
            n_vocab_phone_shengdiao=300
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.n_vocab_note_pitch = n_vocab_pitch

        self.emb_phoneme = nn.Embedding(n_vocab, hidden_channels)
        self.emb_note_pitch = nn.Embedding(n_vocab_pitch, hidden_channels)

        self.emb_phone_shengdiao = nn.Embedding(n_vocab_phone_shengdiao, hidden_channels)

        self.emb_note_duration = nn.Linear(in_features=1, out_features=hidden_channels)

        nn.init.normal_(self.emb_phoneme.weight, 0.0, hidden_channels ** -0.5)
        nn.init.normal_(self.emb_note_pitch.weight, 0.0, hidden_channels ** -0.5)
        # nn.init.normal_(self.emb_note_duration.weight, 0.0, hidden_channels ** -0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, phonemes, note_pitch_id, note_duration, lengths, phone_shengdiao):
        emb_phoneme = self.emb_phoneme(phonemes)
        emb_note_pitch = self.emb_note_pitch(note_pitch_id)
        note_duration_exp = torch.unsqueeze(note_duration, dim=-1)
        emb_note_duration = self.emb_note_duration(note_duration_exp)

        # 添加声调embedding
        emb_phone_shengdiao = self.emb_phone_shengdiao(phone_shengdiao)

        emb = (emb_phoneme + emb_note_duration + emb_note_pitch + emb_phone_shengdiao) * math.sqrt(self.hidden_channels) # [b, t, h]

        x = torch.transpose(emb, 1, -1) # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        y = x * x_mask

        return y, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
            channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            n_flows=4,
            gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.spec_flows = nn.ModuleList()
        for i in range(n_flows):
            self.spec_flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.spec_flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.spec_flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.spec_flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    '''
        解码器,将声谱图转换为隐藏特征,网络架构如下：
        类似WaveNet的扩张卷积（加残差连接）
        输入：声谱图编码,
        输出：预测的隐藏特征, 均值,对数方差,mask
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
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask # torch.exp(logs) torch.sigmoid(logs + 2)
        return z, m, logs, x_mask, x


class Generator(torch.nn.Module):
        '''
        生成器生成音频采样,包括以下模块：
        上采样模块：通过反卷积将特征进行上采样；
        扩张卷积加普通卷积组成的ResBlocks
        输出映射层：通道映射为1,输出预测音频采样

        '''
        def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
                super(Generator, self).__init__()
                self.num_kernels = len(resblock_kernel_sizes)
                self.num_upsamples = len(upsample_rates)
                self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
                resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

                self.ups = nn.ModuleList()
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                        self.ups.append(weight_norm(
                                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                                                k, u, padding=(k-u)//2)))

                self.resblocks = nn.ModuleList()
                for i in range(len(self.ups)):
                        ch = upsample_initial_channel//(2**(i+1))
                        for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                                self.resblocks.append(resblock(ch, k, d))

                self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
                self.ups.apply(init_weights)

                if gin_channels != 0:
                        self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x, g=None):
                x = self.conv_pre(x)
                if g is not None:
                    x = x + self.cond(g)

                for i in range(self.num_upsamples):
                        x = F.leaky_relu(x, modules.LRELU_SLOPE)
                        x = self.ups[i](x)
                        xs = None
                        for j in range(self.num_kernels):
                                if xs is None:
                                        xs = self.resblocks[i*self.num_kernels+j](x)
                                else:
                                        xs += self.resblocks[i*self.num_kernels+j](x)
                        x = xs / self.num_kernels
                x = F.leaky_relu(x)
                x = self.conv_post(x)
                x = torch.tanh(x)

                return x

        def remove_weight_norm(self):
                print('Removing weight norm...')
                for l in self.ups:
                        remove_weight_norm(l)
                for l in self.resblocks:
                        l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
                super(DiscriminatorP, self).__init__()
                self.period = period
                self.use_spectral_norm = use_spectral_norm
                norm_f = weight_norm if use_spectral_norm == False else spectral_norm
                self.convs = nn.ModuleList([
                        norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                        norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                        norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                        norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                        norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
                ])
                self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        def forward(self, x):
                fmap = []

                # 1d to 2d
                b, c, t = x.shape
                if t % self.period != 0: # pad first
                        n_pad = self.period - (t % self.period)
                        x = F.pad(x, (0, n_pad), "reflect")
                        t = t + n_pad
                x = x.view(b, c, t // self.period, self.period)

                for l in self.convs:
                        x = l(x)
                        x = F.leaky_relu(x, modules.LRELU_SLOPE)
                        fmap.append(x)
                x = self.conv_post(x)
                fmap.append(x)
                x = torch.flatten(x, 1, -1)

                return x, fmap


class DiscriminatorS(torch.nn.Module):
        def __init__(self, use_spectral_norm=False):
                super(DiscriminatorS, self).__init__()
                norm_f = weight_norm if use_spectral_norm == False else spectral_norm
                self.convs = nn.ModuleList([
                        norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                        norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                        norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                        norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                        norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                        norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
                ])
                self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

        def forward(self, x):
                fmap = []

                for l in self.convs:
                        x = l(x)
                        x = F.leaky_relu(x, modules.LRELU_SLOPE)
                        fmap.append(x)
                x = self.conv_post(x)
                fmap.append(x)
                x = torch.flatten(x, 1, -1)

                return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
        def __init__(self, use_spectral_norm=False):
                super(MultiPeriodDiscriminator, self).__init__()
                periods = [2,3,5,7,11]

                discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
                discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
                self.discriminators = nn.ModuleList(discs)

        def forward(self, y, y_hat):
                y_d_rs = []
                y_d_gs = []
                fmap_rs = []
                fmap_gs = []
                for i, d in enumerate(self.discriminators):
                        y_d_r, fmap_r = d(y)
                        y_d_g, fmap_g = d(y_hat)
                        y_d_rs.append(y_d_r)
                        y_d_gs.append(y_d_g)
                        fmap_rs.append(fmap_r)
                        fmap_gs.append(fmap_g)

                return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
        生成音频采样,包括以下模块：
        文本编码器：输入音素序列,生成h_text, mean_text, log(seita_text)
        时长预测模块：输入noise和h_text,输出预测的音素时长；
        声谱编码器：将声谱图转换为隐藏特征,输入：声谱图编码,输出：预测的隐藏特征, 均值,对数方差,mask
        可逆流模块：输入均值、方差,输出采样
        解码器：生成器生成音频采样,输入：声谱图隐藏表征,输出：预测音频采样；
    """

    def __init__(self, 
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        n_vocab_note=100,
        **kwargs):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.lr = LengthRegulator()

        self.use_sdp = use_sdp

        # 文本编码器
        self.text_encoder = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            n_vocab_pitch=n_vocab_note,
        )
        self.text_encoder.cuda(0)
        # 解码器
        self.decoder = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        # 时长预测模块
        # if use_sdp:
        #     self.duration_predictor = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        # else:
        #     self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        # # 说话人编码
        # if n_speakers > 1:
        #     self.emb_speaker = nn.Embedding(n_speakers, gin_channels)

        # VITSinger 添加模块
        # 编码器后处理模块
        self.text_post_net = TextPostNet(
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            f0_predictor_layers=6,  # 6
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            out_channels=inter_channels,
            n_layers_frame_prior=4,  # 4
        )

        # # # 音素预测模块
        # self.phoneme_predictor = PhonemePredictor(
        #     hidden_channels=hidden_channels,
        #     filter_channels=filter_channels,
        #     n_heads=n_heads,
        #     n_layers=4,
        #     kernel_size=kernel_size,
        #     p_dropout=p_dropout,
        #     n_vocab=n_vocab,
        # )

        # 声谱编码器
        self.spec_encoder = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

        # 流模块
        self.spec_flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # 声谱解码器
        self.spec_decoder = SpecDecoder(inter_channels, spec_channels, hidden_channels, 5, 1, 16,
                                             gin_channels=gin_channels)

        self.spec_proj = nn.Conv1d(in_channels=spec_channels, out_channels=inter_channels, kernel_size=1)

    def forward(self, x, x_lengths, y, y_lengths, note_pitch_id, note_lengths, f0_score, phone_durations, phone_shengdiao, sid=None):
        h_text, x_mask = self.text_encoder(
            phonemes=x,
            note_pitch_id=note_pitch_id,
            note_duration=note_lengths,
            lengths=x_lengths,
            phone_shengdiao=phone_shengdiao,
        )

        # 音素发音时长预测
        phoneme_length_predict = self.duration_predictor(h_text, x_mask)

        # 长度规整
        h_text_lr, mel_len = self.lr(h_text, phone_durations)

        # 输出添加隐藏状态h_spec,用于计算ctc loss
        z, m_q, logs_q, y_mask, h_spec = self.spec_encoder(y, y_lengths)
        z_p = self.spec_flow(z, y_mask)

        # 后处理网络预测均值、方差
        m_p, logs_p, log_f0, mel_predict = self.text_post_net(h_text_lr, y_lengths, f0_score, y_mask)

        # 添加反向z，用于计算反向kl loss
        z_p_hat = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z_hat = self.spec_flow(z_p_hat, y_mask, reverse=True)
        # if self.n_speakers > 0:
        #     g = self.emb_speaker(sid).unsqueeze(-1) # [b, h, 1]
        # else:
        #     g = None
        # phonemes_predict = self.phoneme_predictor(h_spec, y_mask)

        spec_predict = self.spec_decoder(z, y_lengths)
        spec_hidden = self.spec_proj(spec_predict)

        # 音频切片，训练解码器
        z_slice, ids_slice = commons.rand_slice_segments(spec_hidden, y_lengths, self.segment_size)
        o = self.decoder(z_slice)

        return o, phoneme_length_predict, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), \
               log_f0, spec_predict, z_hat, mel_predict

    def infer(self, x, x_lengths, note_pitch_id, note_duration, phone_shengdiao, ph_dur=None, lf0=None, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        h_text, x_mask = self.text_encoder(
            phonemes=x,
            note_pitch_id=note_pitch_id,
            note_duration=note_duration,
            lengths=x_lengths,
            phone_shengdiao=phone_shengdiao
        )
        # if self.n_speakers > 0:
        #     g = self.emb_speaker(sid).unsqueeze(-1) # [b, h, 1]
        # else:
        #     g = None

        # 音素发音时长预测
        if ph_dur is None:
            phoneme_length_predict = self.duration_predictor(h_text, x_mask)
            phoneme_length = phoneme_length_predict * note_duration
            phone_durations = torch.floor(phoneme_length / 0.0116).long()
        else:
            phone_durations = ph_dur

        # 长度规整
        h_text_lr, mel_len = self.lr(h_text, phone_durations)

        y_mask = torch.unsqueeze(commons.sequence_mask(mel_len, h_text_lr.size(2)), 1).to(x.dtype)

        g = None
        # 后处理网络预测均值、方差
        m_p, logs_p, log_f0, mel_predict = self.text_post_net(h_text_lr, x_lengths, x_mask=y_mask)

        if lf0 is not None:
            log_f0 = lf0
        # m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        # logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.spec_flow(z_p, y_mask, g=g, reverse=True)

        # 添加声谱解码
        spec_predict = self.spec_decoder(z, mel_len)
        spec_hidden = self.spec_proj(spec_predict)

        o = self.decoder((spec_hidden * y_mask)[:,:,:max_len], g=g)
        return o, y_mask, (z, z_p, m_p, logs_p), log_f0, spec_predict, mel_predict

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_speaker(sid_src).unsqueeze(-1)
        g_tgt = self.emb_speaker(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask, _ = self.spec_encoder(y, y_lengths, g=g_src)
        z_p = self.spec_flow(z, y_mask, g=g_src)
        z_hat = self.spec_flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)


    def infer_fake(self, spec, spec_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        z, m_q, logs_q, y_mask, h_spec = self.spec_encoder(spec, spec_lengths)

        y_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype)

        g = None
        o = self.decoder((z * y_mask)[:,:,:max_len], g=g)
        return o, y_mask, z

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_speaker(sid_src).unsqueeze(-1)
        g_tgt = self.emb_speaker(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask, _ = self.spec_encoder(y, y_lengths, g=g_src)
        z_p = self.spec_flow(z, y_mask, g=g_src)
        z_hat = self.spec_flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len=None):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils.pad(output, max_len)
        else:
            output = utils.pad(output)

        # [B, T, C] -> [B, C, T]
        output = torch.transpose(output, 1, 2)
        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        # [C, T] -> [T, C]
        batch = torch.transpose(batch, 0, 1)

        # [T, C] -> [T*n, C]
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len=None):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len
