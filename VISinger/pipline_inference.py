# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/6/16
# Description:


import os
import torch
import time

from VISinger import utils
from VISinger.models import SynthesizerTrn
from VISinger.cpop_text_preprocessor import *

from scipy.io.wavfile import write
import numpy as np
import scipy

from VISinger.text_normalizers.text_normalizer import TextNormalizer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SAMPLE_RATE = 22050
SHENG_MU = set(['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'y', 'w', 'zh',
                 'ch', 'sh', 'r'])


class Synthesiser:
    def __init__(self, pretrained_model_path, hps_path="./configs/ours.json"):
        self.hps = utils.get_hparams_from_file(hps_path)
        self.net_g = SynthesizerTrn(
            100,
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        _ = self.net_g.eval()

        _ = utils.load_checkpoint(pretrained_model_path, self.net_g, None)
        self.phone_encoder = build_phone_encoder("./datasets")
        self.phone_encoder_shengdiao = build_phone_encoder("./datasets", fname="phone_shengdiao_set.json")

        monosyllable_dict_dir = "./VISinger/text_normalizers/data/monosyllable.csv"
        polyphone_dict_dir = "./VISinger/text_normalizers/data/polyphone.csv"
        self.text_normalizer = TextNormalizer(monosyllable_dict_dir, polyphone_dict_dir)

    def process_input_data(self, text, notes, note_lengths):
        ph_str_result = ""
        pause_poses = []
        new_notes = []
        new_note_lengths = []

        pos = 0
        for text in text.split("[sep]"):
            pause_poses.append(len(text))
            phonemes = self.text_normalizer.normalize(text, word_parse=False).replace(".", "").strip()
            ph_seq = []
            phonemes = phonemes.split(" ")
            for i in range(len(phonemes)):
                ph = phonemes[i]
                if ph[:2] in SHENG_MU:
                    ph_seq += [ph[:2], ph[2:]]
                    new_notes += [notes[pos]] * 2
                    new_note_lengths += [note_lengths[pos]] * 2
                elif ph[:1] in SHENG_MU:
                    ph_seq += [ph[:1], ph[1:]]
                    new_notes += [notes[pos]] * 2
                    new_note_lengths += [note_lengths[pos]] * 2
                else:
                    ph_seq += [ph]
                    new_notes += [notes[pos]]
                    new_note_lengths += [note_lengths[pos]]
                pos += 1

            ph_str = " ".join(ph_seq)
            ph_str_result += ph_str + " SP "
            new_notes += ["rest"]
            new_note_lengths += ["0.6"]

        ph_str_result = ph_str_result[:-3] + "AP"
        return ph_str_result, new_notes, new_note_lengths


    def get_text(self, text):
        metas = text.split("|")
        text = metas[1].replace(" ", "")
        notes = metas[2]
        note_lengths = metas[3]

        notes = notes.split(" ")
        note_lengths = note_lengths.split(" ")
        phoneme_shengdiao, notes, note_lengths = self.process_input_data(text, notes, note_lengths)

        print("phoneme_shengdiao:", phoneme_shengdiao)
        phonemes = phoneme_shengdiao.replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "")

        phones = format_phone(self.phone_encoder, phonemes)

        notes = format_note(notes)
        note_lengths = np.asarray([float(x) for x in note_lengths]).astype(np.float)
        phoneme_shengdiao = format_phone(self.phone_encoder_shengdiao, phoneme_shengdiao)
        print(phoneme_shengdiao)

        assert len(phones) == len(notes) == len(note_lengths) == len(phoneme_shengdiao), \
            f"len(phones)={len(phones)} == len(notes)={len(notes)} == len(note_lengths)={len(note_lengths)} == len(phoneme_shengdiao) = {len(phoneme_shengdiao)}"
        phones = torch.LongTensor(phones)
        notes = torch.LongTensor(notes)
        note_lengths = torch.FloatTensor(note_lengths)
        phoneme_shengdiao = torch.LongTensor(phoneme_shengdiao)

        return phones, notes, note_lengths, phoneme_shengdiao


    def syn(self, text, wave_save_path=None):
        phones, notes, note_lengths, phoneme_shengdiao = self.get_text(text)
        with torch.no_grad():
            phones_cuda = phones.cuda().unsqueeze(0)
            notes = notes.cuda().unsqueeze(0)
            note_lengths = note_lengths.cuda().unsqueeze(0)
            phoneme_shengdiao = phoneme_shengdiao.cuda().unsqueeze(0)

            x_lengths = torch.LongTensor([phones.size(0)]).cuda()
            audio = self.net_g.infer(phones_cuda, x_lengths, notes, note_lengths, noise_scale=1., length_scale=1, phone_shengdiao=phoneme_shengdiao)[0][0,0].data.cpu().float().numpy()

            if wave_save_path is not None:
                audio *= 32767 / max(0.01, np.max(np.abs(audio)))
                write(wave_save_path, SAMPLE_RATE, audio.astype(np.int16))

            return audio


def main():
    save_dir = "./syn_outputs"
    os.makedirs(save_dir, exist_ok=True)
    pretrained_model_path = "./logs/test-0921-finetune-v3/G_235000.pth"
    syn = Synthesiser(pretrained_model_path)

    texts = [
        "今夜特别漫长[sep]有个号码一直被存放|G4 G4 D4 D4 D4 D4 D♯4 D4 D4 D4 C4 C4 C4 C4 C4|0.24 0.24 0.24 0.24 0.24 0.72 0.48 0.24 0.24 0.24 0.24 0.24 0.24 0.24 0.72",
    ]

    sname  = "测试"
    wavs = []
    for i, text in enumerate(texts):
        text = "test" + str(i) + "|" + text
        print("Synthesising text:%s" % text)
        t0 = time.time()
        wave_save_path = os.path.join(save_dir, text.split("|")[0] + ".wav")
        wav = syn.syn(text, wave_save_path)
        print("Time cost:%.3f" % (time.time() - t0))
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavs = np.concatenate([wavs, wav])
        # print("wave 耗时：%.3f" % (time.time() - t1))
        print("合成耗时：%.3f" % (time.time() - t0))

    scipy.io.wavfile.write(f'{save_dir}/{sname}.wav', 22050, wavs.astype(np.int16))
    print(f"歌声合成完成，保存路径：{save_dir}/{sname}.wav")

    print("Done!")


if __name__ == "__main__":
    main()
