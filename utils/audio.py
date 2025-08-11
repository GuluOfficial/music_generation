# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/7/6
# Description:
import librosa
import librosa.filters
import numpy as np
import scipy
from scipy import io
from scipy.io import wavfile
import os


class Audio:
    """
    音频处理
    """

    def __init__(self, samplerate=16000):
        self.samplerate = samplerate

    def load_wave(self, wave_path):
        """
        加载音频
        :param wave_path:
        :return:
        """
        return librosa.core.load(wave_path, sr=self.samplerate)[0]

    # 保存音频
    def save_wave(self, wave, wave_save_path):
        """
        保存音频
        :param wave:
        :param wave_save_path:
        """
        wave *= 32767 / max(0.01, np.max(np.abs(wave)))
        scipy.io.wavfile.write(wave_save_path, self.samplerate, wave.astype(np.int16))


def change_audio_samplerate(aduio_path, audio_save_path, target_sample_rate=16000):
    """
    修改音频格式
    :param aduio_path:
    :param target_sample_rate:
    :return:
    """
    tool = Audio(target_sample_rate)
    wave = tool.load_wave(aduio_path)
    new_path = os.path.join(audio_save_path)
    tool.save_wave(wave, new_path)

    return new_path
