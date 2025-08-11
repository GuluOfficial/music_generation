# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2020/11/16
# Description: 一些音频处理函数（包括stft、MFC等）


import librosa
import librosa.filters
import numpy as np
import random
import scipy
from scipy import io
from scipy import signal
from scipy.io import wavfile
import parselmouth

from VISinger.utils.config_loader import ConfigLoader
from VISinger.utils.pitch_utils import f0_to_coarse


class Audio:
    def __init__(self, config_file_path):
        confing_loader = ConfigLoader(config_file_path)
        self.hparams = confing_loader.get_preprocessing_params()
        self.mel_basis = self.build_mel_basis()

    # 加载音频
    def load_wave(self, wave_path):
        # load(wave_path, res_type='kaiser_fast', sr=self.hparams["sample_rate"])[0]#
        return librosa.core.load(wave_path, sr=self.hparams["sample_rate"])[0]

    # 保存音频
    def save_wave(self, wave, wave_save_path):
        wave *= 32767 / max(0.01, np.max(np.abs(wave)))
        scipy.io.wavfile.write(wave_save_path, self.hparams["sample_rate"], wave.astype(np.int16))

    # 音频长度规整为固定长度
    def norm_audio_length(self, wav, norm_len):
        len_wav = len(wav)
        if len_wav < norm_len:
            dup_num = int(norm_len / len_wav) + 1
            wav = np.tile(wav, dup_num)[:norm_len]
        else:
            start_pos = random.randint(0, len_wav - norm_len)
            wav = wav[start_pos: start_pos + norm_len]
        return wav

    # 音频预加重
    def preemphasis(self, x):
        return scipy.signal.lfilter([1, -self.hparams["preemphasis"]], [1], x)

    # 音频去预加重
    def inv_preemphasis(self, x):
        return scipy.signal.lfilter([1], [1, -self.hparams["preemphasis"]], x)

    # 生成线性谱
    def linear_spectrogram(self, x):
        hp = self.hparams
        x = self.preemphasis(x)
        s = self.stft(x)
        s = self.amp2db(np.abs(s)) - hp["env_level_db"]
        return s

    def build_mel_basis(self):
        hp = self.hparams
        return librosa.filters.mel(self.hparams["sample_rate"], n_fft=hp["fft_size"], n_mels=hp["n_mel_channels"], fmin=50, fmax=7600)

    # 生成梅尔谱
    def mel_spectrogram(self, x):
        hp = self.hparams
        # mel_s = librosa.feature.melspectrogram(x, sr=hp["sample_rate"],  n_mels=hp["n_mel_channels"],
        #                                        n_fft=hp["fft_size"], hop_length=hp["hop_size"])
        # mel_s = librosa.amplitude_to_db(mel_s)
        x = self.preemphasis(x)
        linear_s = self.stft(x)
        mel_basis = self.mel_basis
        mel_s = np.dot(mel_basis, np.abs(linear_s))
        mel_s = self.amp2db(mel_s)
        return self.normalize(mel_s)
        #return mel_s

    # stft 变换
    def stft(self, x):
        hp = self.hparams
        return librosa.stft(y=x, n_fft=hp["fft_size"], hop_length=hp["hop_size"], win_length=hp["win_size"])

    def istft(self, y):
        hp = self.hparams
        return librosa.istft(y, hop_length=hp["hop_size"], win_length=hp["win_size"])

    # 分贝到能量转换
    def amp2db(self, x):
        hp = self.hparams
        min_level = np.exp(hp["min_db"] / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    # 能量到分贝转换
    def db2amp(self, x):
        return np.power(10.0, x * 0.05)

    # 经验归一化处理（不用）
    def normalize(self, x):
        hp = self.hparams
        #max_db = hp["max_db"]
        min_db = hp["min_db"]

        return 8 * ((x-min_db) / - min_db) - 4

    # 经验归一化处理（不用）
    def denormalize(self, x):
        hp = self.hparams
        min_db = hp["min_db"]

        x = np.clip(x, -4, 4)
        x = (x + 4) / 8
        return x * (- min_db) + min_db

    # 去除前后静音
    def trim_silence(self, x):
        hp = self.hparams
        x, index = librosa.effects.trim(x, top_db=hp["top_db"])
        return x

    # 快速去除前后静音
    def trim_silence_fast(self, x, thread_hold=0.02, min_sil_sec=0.1):
        len_wav = len(x)
        hp = self.hparams
        sr = hp["sample_rate"]

        win_size = int(min_sil_sec * sr)
        hop_size = int(0.5 * win_size)
        i = 0
        while i < len_wav - 2 * win_size:
            if np.max(np.abs(x[i:i+win_size])) < thread_hold:
                i += hop_size
            else:
                break
        start_pos = i

        i = len_wav
        while i > 2 * win_size:
            if np.max(np.abs(x[i-win_size:i])) < thread_hold:
                i -= hop_size
            else:
                break
        end_pos = i

        return x[start_pos:end_pos]

    # 简单去噪
    def denoise(self, x):
        x = scipy.signal.wiener(x)
        return x

    def mel_to_linear(self, mel_spectrogram):
        inv_mel_basis = np.linalg.pinv(self.mel_basis)
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self.denormalize(mel_spectrogram)

        S = self.mel_to_linear(self.db2amp(D))
        return self.inv_preemphasis(self.griffin_lim(S ** self.hparams["power"]))

    def griffin_lim(self, S):
        '''librosa implementation of Griffin-Lim
        Based on https://github.com/librosa/librosa/issues/434
        '''
        hp = self.hparams
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self.istft(S_complex * angles)
        for i in range(hp["griffin_lim_iters"]):
            angles = np.exp(1j * np.angle(self.stft(y)))
            y = self.istft(S_complex * angles)
        return y

    # 获取基频
    def f0(self, wav_path, spec_len, time_step=0.0125):
        hp = self.hparams
        snd = parselmouth.Sound(wav_path)
        f0 = snd.to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6,
        pitch_floor=hp["f0_min"], pitch_ceiling=hp["f0_max"]).selected_array['frequency']

        if hp['hop_size'] == 128:
            pad_size = 4
        elif hp['hop_size'] == 256:
            pad_size = 2
        else:
            assert False

        # zsk change 2022-06-15

        lpad = 0#pad_size * 2 - 1
        rpad = spec_len - len(f0) - lpad

        assert rpad >= 0, f"{spec_len}, {lpad}, {len(f0)}"
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
        # mel and f0 are extracted by 2 different libraries. we should force them to have the same length.
        # Attention: we find that new version of some libraries could cause ``rpad'' to be a negetive value...
        # Just to be sure, we recommend users to set up the same environments as them in requirements_auto.txt (by Anaconda)
        delta_l = spec_len - len(f0)
        assert np.abs(delta_l) <= 8
        if delta_l > 0:
            f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
        f0 = f0[:spec_len]
        pitch_coarse = f0_to_coarse(f0)
        return np.log(f0+1), pitch_coarse


if __name__ == "__main__":

    config_path = "../configs/config_tacotron2.yaml"
    audio_tool = Audio(config_path)

    wav = audio_tool.load_wave(r"F:\data\biaobei\Wave\000011.wav")
    wav = audio_tool.trim_silence(wav)
    mel = audio_tool.mel_spectrogram(wav)
    print(np.max(mel), np.min(mel))

    #mel = np.load(r"F:\data\biaobei\training\mels\000001_mel.npy")
    wav = audio_tool.inv_mel_spectrogram(mel)
    wav = audio_tool.save_wave(wav, wave_save_path=r"F:\data\test.wav")
