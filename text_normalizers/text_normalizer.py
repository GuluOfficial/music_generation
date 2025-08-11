# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/3/1
# Description:文本预处理
import sys

sys.path.append("/home/zsk/code/TensorFlowTTS-master")
import jieba

from VISinger.text_normalizers.number_normalizer import NumberNormalizer
from VISinger.text_normalizers.symbol_normalizer import SymbolNormalizer
from VISinger.text_normalizers.phoneme_generator import PhonemeGenerator


class TextNormalizer:
    """
    文本预处理脚本
    """
    def __init__(self, monosyllable_dict_dir, polyphone_dict_dir, jieba_dict_dir=None):
        self._number_normalizer = NumberNormalizer()
        self._symbol_normalizer = SymbolNormalizer()
        self._phoneme_generator = PhonemeGenerator(monosyllable_dict_dir, polyphone_dict_dir)
        self._jieba_dict = None#self.load_jieba_dict(jieba_dict_dir) if jieba_dict_dir is not None else None

    def load_jieba_dict(self, jieba_dict_dir):
        try:
            jieba.load_userdict(jieba_dict_dir)
        except:
            print("jieba dict load fail. dict dir:", jieba_dict_dir)
            return {}

    def normalize(self, text, word_parse=False):
        """
        文本规范化处理，生成音素
        :param text:
        :return:
        """
        text = text.lower()
        text = self._number_normalizer.normalize(text)
        text = self._symbol_normalizer.normalize(text)
        if word_parse:
            text = "#".join(jieba.cut(text, cut_all=False)) # if self._jieba_dict is not None else text  # 结巴分词
        text = self._phoneme_generator.grapheme2phoneme(text)
        if word_parse:
            text = self._symbol_normalizer.normalize_sharp(text)

        return text


# 测试
if __name__ == "__main__":
    monosyllable_dict_dir = "./data/monosyllable.csv"
    polyphone_dict_dir = "./data/polyphone.csv"
    jieba_dict_dir = "./data/userdict.txt"
    test_text = "长亭外SPAP古道边"

    normalizer = TextNormalizer(monosyllable_dict_dir, polyphone_dict_dir, jieba_dict_dir)
    phonemes = normalizer.normalize(test_text)

    print(phonemes)
