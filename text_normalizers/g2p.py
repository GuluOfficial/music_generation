# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/3/9
# Description:
import sys
root_dir = "./text_normalizers"
sys.path.append(root_dir)
from text_normalizers.text_normalizer import *


def main():
    monosyllable_dict_dir = root_dir + "/data/monosyllable.csv"
    polyphone_dict_dir = root_dir + "/data/polyphone.csv"
    jieba_dict_dir = root_dir + "/data/userdict.txt"

    g_file_path = r"/data/zsk/svs/cpop/mixed-data/train.txt"
    p_file_path = r"/data/zsk/svs/cpop/mixed-data/train_py.txt"

    split_symbol = "|"

    normalizer = TextNormalizer(monosyllable_dict_dir, polyphone_dict_dir, jieba_dict_dir)

    new_lines = []
    with open(g_file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            metadata = line.strip().split(split_symbol)
            fname, text = metadata[0], metadata[1].replace(",", "")
            phonemes = normalizer.normalize(text, word_parse=False)

            phonemes = phonemes.replace(".", "")
            phs = []
            shengmu = set(
                ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'z', 'c', 's', 'y', 'w', 'zh',
                 'ch', 'sh', 'r'])
            for ph in phonemes.split(" "):
                if len(ph) < 1:
                    continue
                if ph[:2] in shengmu:
                    phs.append(ph[:2])
                    phs.append(ph[2:])
                elif ph[0] in shengmu:
                    phs.append(ph[0])
                    phs.append(ph[1:])
                else:
                    phs.append(ph)
            phonemes = " ".join(phs)
            new_line = fname + "|" + phonemes + "\n"
            new_line = new_line.replace(".", "")
            new_lines.append(new_line)

    with open(p_file_path, 'wt', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)


if __name__ == "__main__":
    main()
