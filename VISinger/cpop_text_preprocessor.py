# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/5/6
# Description: cpop 标注数据预处理

import os
import numpy as np
import traceback
import librosa
import json

from VISinger.utils.text_encoder import TokenTextEncoder


def build_phone_encoder(data_dir="./VISinger", fname='phone_set.json'):
    phone_list_file = os.path.join(data_dir, fname)
    phone_list = json.load(open(phone_list_file))
    return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')


# 获取音素id字典
def get_phone_id_dict(phone_dict_file="../datasets/transcriptions.txt"):
    phone_id_dict = {
        '-':0, 'an': 1, 'sh': 2, 'ou': 3, 't': 4, 'ing': 5, 'z': 6, 'ai': 7, 'w': 8, 'o': 9, 'f': 10,
        'a': 11, 'd': 12, 'uan': 13, 'e': 14, 'SP': 15, 'zh': 16, 'i': 17, 'j': 18, 'ian': 19, 'AP': 20,
        'r': 21, 'u': 22, 'h': 23, 'un': 24, 'ong': 25, 'ie': 26, 'ang': 27, 'uang': 28, 'y': 29, 'x': 30,
         'v': 31, 'm': 32, 'ei': 33, 'ui': 34, 'ao': 35, 'en': 36, 'iao': 37, 'iang': 38, 'ch': 39, 'p': 40,
        'n': 41, 'van': 42, 'l': 43, 'q': 44, 've': 45, 'eng': 46, 'k': 47, 'b': 48, 'in': 49, 'iong': 50,
        's': 51, 'er': 52, 'uo': 53, 'c': 54, 'vn': 55, 'iu': 56, 'ia': 57, 'uai': 58, 'ua': 59, 'g': 60,
    }

    # phone_id_dict = {}
    # with open(phone_dict_file, 'rt', encoding='utf-8') as f:
    #     idx = 0
    #     for l in f:
    #         l = l.strip()
    #         phones = l.split("|")[2].split(" ")
    #         for phone in phones:
    #             if phone not in phone_id_dict:
    #                 phone_id_dict[phone] = idx
    #                 idx += 1

    return phone_id_dict


def get_note_id_dict(note_file="../datasets/transcriptions.txt"):
    # note_id_dict = {
    #     '~':  0,  'A1': 1, 'A2': 2, 'A3': 3, 'A4': 4, 'A5': 5, 'A6': 6, 'A7': 7, 'A8': 8, 'B1': 9, 'B2': 10, 'B3': 11,
    #     'B4': 12, 'B5': 13, 'B6': 14, 'B7': 15, 'B8': 16, 'C1': 17, 'C2': 18, 'C3': 19, 'C4': 20, 'C5': 21, 'C6': 22,
    #     'C7': 23, 'C8': 24, 'D1': 25, 'D2': 26, 'D3': 27, 'D4': 28, 'D5': 29, 'D6': 30, 'D7': 31, 'D8': 32, 'E1': 33,
    #     'E2': 34, 'E3': 35, 'E4': 36, 'E5': 37, 'E6': 38, 'E7': 39, 'E8': 40, 'F1': 41, 'F2': 42, 'F3': 43, 'F4': 44,
    #     'F5': 45, 'F6': 46, 'F7': 47, 'F8': 48, 'G1': 49, 'G2': 50, 'G3': 51, 'G4': 52, 'G5': 53, 'G6': 54, 'G7': 55,
    #     'G8': 56, 'rest': 57,
    # }

    note_id_dict = {}
    with open("midi-note.scp", 'rt', encoding='utf-8') as f:
        idx = 0
        for l in f:
            l = l.strip()
            pair = l.split("\t")
            note_id_dict[pair[1]] = pair[0]

    return note_id_dict


# 将phone进行id化，以便进行embedding
def format_phone(phone_encoder, phones):#phones, phone_idx_dict):
    # phone_ids = [0] * len(phones)
    # for i, phone in enumerate(phones):
    #     phone_id = phone_idx_dict[phone]
    #     phone_ids[i] = phone_id
    phone_ids = phone_encoder.encode(phones)

    return phone_ids


# 将note发音进行id化，以便进行embedding
def format_note(notes, note_idx_dict=None):
    # note_ids = [0] * len(notes)
    # for i, note in enumerate(notes):
    #     if note.find("/") > -1:
    #         note = note[:note.find("/")].replace("#", "")
    #     note_id = note_idx_dict[note]
    #     note_ids[i] = note_id
    note_ids = [librosa.note_to_midi(x.split("/")[0].replace("5", "4").replace("6", "4")) if x != 'rest' else 0 for x in notes]
    return note_ids


# 将note发音时长进行id化，以便进行embedding
def format_note_duration(durations, win_size=0.005):
    durations = [float(d) for d in durations]
    N = 100000
    win_size = int(N * win_size)

    duration_ids = [0] * len(durations)
    for i, duration in enumerate(durations):
        d = int(N * duration)

        idx = int((d + win_size//2) // win_size)

        duration_ids[i] = idx

    return duration_ids


# 将音素时长按帧长度进行量化，作为模型预测监督数据
def format_phone_duration(durations, frame_len=0.0125):
    durations = [float(d) for d in durations]
    sum_len = sum(durations)

    K = 10e6
    frame_len = int(frame_len * K)

    N = int(sum_len * K) // frame_len
    if int(sum_len * K) % frame_len > 0:
        N += 1

    M = len(durations)
    duration_n_frames = [0] * M
    bias = 0

    for i, d in enumerate(durations):
        d = int(K * d)
        d += bias
        if d < frame_len:
            n = 1
            bias = d - frame_len
        else:
            n = d // frame_len
            left = d % frame_len
            if left / frame_len > 0.5:
                n += 1
                bias = left - frame_len
            else:
                bias = left
        duration_n_frames[i] = n
    if bias > 0:
        duration_n_frames[-1] += 1

    if sum(duration_n_frames) != N:
        duration_n_frames[-1] -= 1

    assert sum(duration_n_frames) == N, "Duration format error! N=%d, sum d=%d" % (N, sum(duration_n_frames))
    return duration_n_frames


def main(data_path, sava_dir="./outputs"):
    with open(data_path, 'rt', encoding='utf-8') as f:
        for l in f:
            try:
                l = l.strip()
                metas = l.split("|")

                base_name = metas[0]
                phones = metas[2]
                notes = metas[3]
                note_durations = metas[4]
                phone_durations = metas[5]

                phones = format_phone(phones.split(" "), get_phone_id_dict())
                notes = format_note(notes.split(" "), get_note_id_dict())
                note_durations = format_note_duration(note_durations.split(" "))
                phone_durations = format_phone_duration(phone_durations.split(" "))

                np.save(os.path.join(sava_dir, base_name+"-phone-ids.npy"), np.asarray(phones).astype(np.int32), allow_pickle=False)
                np.save(os.path.join(sava_dir, base_name + "-note-ids.npy"), np.asarray(notes).astype(np.int32), allow_pickle=False)
                np.save(os.path.join(sava_dir, base_name + "-note-duration-ids.npy"), np.asarray(note_durations).astype(np.int32), allow_pickle=False)
                np.save(os.path.join(sava_dir, base_name + "-phone-duration.npy"), np.asarray(phone_durations).astype(np.int32), allow_pickle=False)
            except:
                print("Record:%s error!" % l)
                traceback.print_exc()
                exit(1)

    print("Done!")


if __name__ == "__main__":
    data_path = "../datasets/test.txt"
    sava_dir = "/data/zsk/svs/cpop/training-v3/test"

    os.makedirs(sava_dir, exist_ok=True)

    main(data_path, sava_dir)

