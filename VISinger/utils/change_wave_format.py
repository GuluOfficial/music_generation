# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :pythonProject1
# @File     :change_wave_format
# @Date     :2022/6/5 21:24
# @Author   :ZHSK
# @Email    :skzheng@163.com
# @Describe  :
-------------------------------------------------
"""
import librosa
import os
import scipy
from scipy import io
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm


def change_samplerate(src_dir, save_dir, samplerate=22050):
    fs = [f for f in os.listdir(src_dir) if f.endswith(".wav")]
    for f in tqdm(fs):
        wave_save_path = os.path.join(save_dir, f)
        wave = librosa.core.load(os.path.join(src_dir, f), sr=samplerate)[0]
        wave *= 32767 / max(0.01, np.max(np.abs(wave)))
        scipy.io.wavfile.write(wave_save_path, samplerate, wave.astype(np.int16))


def main():
    src_dir = "/data/zsk/svs/cpop/segments/wavs"
    save_dir = "/data/zsk/svs/cpop/segments/wavs-22k"

    os.makedirs(save_dir, exist_ok=True)

    change_samplerate(src_dir, save_dir)


if __name__ == "__main__":
    main()

