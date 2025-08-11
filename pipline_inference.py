# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/6/16
# Description:
import io
import logging
import torch
import time
from midi2audio import FluidSynth

from VISinger import utils
from VISinger.models import SynthesizerTrn
from VISinger.cpop_text_preprocessor import *

from scipy.io.wavfile import write
import numpy as np
import scipy
import random

from VISinger.text_normalizers.text_normalizer import TextNormalizer
from Melody.melody_generator import get_melody

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

SAMPLE_RATE = 22050
SHENG_MU = set(['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'y', 'w', 'zh',
                 'ch', 'sh', 'r'])
save_dir = "./syn_outputs"
os.makedirs(save_dir, exist_ok=True)
pretrained_visinger_path = "./VISinger/saved_models/G_955000.pth"
# pretrained_visinger_path = "./VISinger/saved_models/G_430000.pth"


class Synthesiser:
    def __init__(self, pretrained_model_path, hps_path="./VISinger/configs/ours.json"):
        self.hps = utils.get_hparams_from_file(hps_path)
        self.net_g = SynthesizerTrn(
            100,
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model).cuda()
        _ = self.net_g.eval()

        _ = utils.load_checkpoint(pretrained_model_path, self.net_g, None)
        self.phone_encoder = build_phone_encoder("./VISinger")
        self.phone_encoder_shengdiao = build_phone_encoder("./VISinger", fname="phone_shengdiao_set.json")

        monosyllable_dict_dir = "./VISinger/text_normalizers/data/monosyllable.csv"
        polyphone_dict_dir = "./VISinger/text_normalizers/data/polyphone.csv"
        self.text_normalizer = TextNormalizer(monosyllable_dict_dir, polyphone_dict_dir)

    def process_input_data(self, text, notes, note_lengths):
        ph_str_result = ""

        new_notes = []
        new_note_lengths = []

        pos = 0
        # for text in text.split("[sep]"):
        text = text[:text.rfind("[sep]")]
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
        ph_str_result += ph_str + " AP"
        new_notes += [notes[-1]]
        new_note_lengths += [note_lengths[-1]]

        return ph_str_result, new_notes, new_note_lengths


    def get_text(self, text):
        metas = text.split("|")
        text = metas[1].replace(" ", "")
        notes = metas[2]
        note_lengths = metas[3]

        notes = notes.split(" ")
        note_lengths = note_lengths.split(" ")
        phoneme_shengdiao, notes, note_lengths = self.process_input_data(text, notes, note_lengths)

        # print("phoneme_shengdiao:", phoneme_shengdiao)
        phonemes = phoneme_shengdiao.replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace("5", "")

        phones = format_phone(self.phone_encoder, phonemes)

        notes = format_note(notes)
        note_lengths = np.asarray([float(x) for x in note_lengths]).astype(np.float)
        phoneme_shengdiao = format_phone(self.phone_encoder_shengdiao, phoneme_shengdiao)
        # print(phoneme_shengdiao)

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


#  解析旋律文件，获取可用信息
def parse_melody(notes):
    pitches = []
    durations = []
    for i, note in enumerate(notes):
        pitch = note.pitch
        pitch = librosa.midi_to_note(pitch)
        pitches.append(pitch)

        if i > 0:
            durations.append( (note.start - notes[i-1].start) / 1000.)

    durations.append((notes[-1].end - notes[-1].start) / 1000.)

    durations = [d for d in durations]
    return pitches, durations


syn = Synthesiser(pretrained_visinger_path)
fluid_synth = FluidSynth("./sound_fonts/MuseScore_General.sf2")
def syn_music(lyrics):
    # texts = [
    #     "今夜特别漫长[sep]有个号码一直被存放|G4 G4 D4 D4 D4 D4 D♯4 D4 D4 D4 C4 C4 C4 C4 C4|0.24 0.24 0.24 0.24 0.24 0.72 0.48 0.24 0.24 0.24 0.24 0.24 0.24 0.24 0.72",
    # ]
    try:
        midi_obj, lyrics = get_melody(lyrics)
    except:
        logging.ERROR("旋律生成出错")
        return None

    lyrics_list = lyrics.split(" ") + ['[sep]']
    start_pos = 0
    end_pos = 0
    notes = midi_obj.instruments[0].notes
    n = len(notes)
    sname  = "测试"
    wavs = []
    delay_time = notes[0].start / 1000. #* 1.56

    i = 0
    while len(lyrics_list) > 0:
        sep_pos = lyrics_list.index("[sep]")
        cur_lyrics_list = lyrics_list[:sep_pos]
        if sep_pos + 1 < n:
            lyrics_list = lyrics_list[sep_pos + 1:]
        else:
            lyrics_list = []
        if len(cur_lyrics_list) < 1:
            break
        end_pos += len(cur_lyrics_list)

        pitches, durations = parse_melody(notes[start_pos:end_pos])
        durations[-1] = durations[-1] + 0.3 + random.random() * 0.1 if durations[-1] < 0.4 else durations[-1] + 0.1 + random.random() * 0.2
        # print(pitches, durations)
        if end_pos < n:
            pitches += ["rest"]
            durations += [random.random() * 0.2 + 0.7]#[(notes[end_pos].start - notes[end_pos-1].end) / 1000.]
        else:
            pitches += ["rest"]
            durations += [1.0]

        durations = " ".join([str(d) for d in durations])
        pitches = " ".join(pitches)
        cur_lyrics = "".join(cur_lyrics_list) + "[sep]"

        text = f"{cur_lyrics}|{pitches}|{durations}"

        text = "test" + str(i) + "|" + text
        print("Synthesising text:%s" % text)
        t0 = time.time()
        wave_save_path = os.path.join(save_dir, text.split("|")[0] + ".wav")
        wav = syn.syn(text, wave_save_path)
        # print("Time cost:%.3f" % (time.time() - t0))
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavs = np.concatenate([wavs, wav])
        # print("wave 耗时：%.3f" % (time.time() - t1))
        print("合成耗时：%.3f" % (time.time() - t0))
        i += 1

    midi_file_path = f'{save_dir}/{sname}.midi'
    # 保存midi文件
    midi_obj.dump(midi_file_path, charset="utf-8")

    midi_wav_file_path = f'{save_dir}/{sname}_midi.wav'

    fluid_synth.midi_to_audio(midi_file_path, midi_wav_file_path)
    wav_midi = librosa.core.load(midi_wav_file_path, sr=SAMPLE_RATE)[0]
    wav_midi *= 32767 / max(0.01, np.max(np.abs(wav_midi)))

    delay_padding = [0.] * int(delay_time * SAMPLE_RATE)
    wav_music = np.concatenate([delay_padding, wavs])

    # print("delay_time:", delay_time)
    for i in range(len(wav_midi)):
        wav_midi[i] = 0.8 * wav_midi[i] + wav_music[i] if i < len(wav_music) else wav_midi[i]

    wav_midi *= 32767 / max(0.01, np.max(np.abs(wav_midi)))
    
    # 加背景声
    wavs = wav_midi

    # 去背景声
    # wavs = wav_music

    wave_file_path = f'{save_dir}/{sname}.wav'

    # 保存歌声
    scipy.io.wavfile.write(wave_file_path, SAMPLE_RATE, wavs.astype(np.int16))
    print(f"歌声合成完成，保存路径：{save_dir}/{sname}.wav")

    out_stream = io.BytesIO()
    scipy.io.wavfile.write(out_stream, SAMPLE_RATE, wavs.astype(np.int16))
    wav_data = out_stream.getvalue()
    out_stream.flush()
    out_stream.close()

    return wav_data


if __name__ == "__main__":
    lyrics = "之江潮起，掀起青春的浪花，趁年华，让我们一起歌唱"
    syn_music(lyrics)
