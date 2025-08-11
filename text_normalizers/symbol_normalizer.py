# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/3/1
# Description: 文本停顿符号规整

from re import compile
from re import sub

# # 号处理
_sharp_pat_1 = compile(r'# ; #')
_sharp_pat_2 = compile(r'# , #')
_sharp_pat_3 = compile(r'# \. #')
_sharp_pat_4 = compile(r'(# )+')


class SymbolNormalizer:
    """
    符号规整
    """
    # 中文符号转换
    cn_symbol_transform_dict = {
        "，": ",", "。": ".", "？": "?", "！": "!",
        "·": "-", "：": ":", "；": ";", "“": "'",
        "”": "'", "{": "{", "}": "}", "【": "[",
        "】": "]", "《": "<", "》": ">", "（": "(",
        "）": ")", "＂": "'", "∙": "", "、": ",",
        "‘": "'", "’": "'", "……": ",", "•": "-",
        "「": "'", "」": "'", "\n": "", " ": ",",
        "—": "-", "．": ".",
        "℃": "摄氏度", "℉": "华氏度", "+": "加",
        "km/h": "千米每小时", "m/s": "米每秒", "119火警": "幺幺九火警",
        "AAAAA风景区": "五A风景区", "AAAA风景区": "四A风景区", "AAA风景区": "三A风景区",
        "AAAAA级": "五A级", "AAAA级": "四A级", "AAA级": "三A级",
    }

    # 英文符号正则化处理
    en_symbol_norm_dict = {
        "-": ";", "_": ",",
        "{": ";", "}": ";",
        "(": ";", ")": ";",
        "[": ";", "]": ";",
        "<": ";", ">": ";",
        "'": ";", "\"": ";",
        ";": ",", ":": ",",
        ",": ",",
        ".": ".",
        "?": ".",
        "!": "."
    }

    pause_symbols = set("-_{}[]()<>'\";:,.?! ")

    def __init__(self):
        pass

    def normalize(self, text):
        """
        规范化文本符号
        :param text:
        :return: normalized text
        """
        text = self.normalize_cn_symbol(text)
        text = self.normalize_en_symbol(text)
        text = self.normalize_multi_symbol(text)
        text = self.normalize_text_tail(text)

        return text

    def normalize_cn_symbol(self, text):
        """
        转换中文符号
        :param text:
        :return:normalized text
        """
        d = self.cn_symbol_transform_dict
        for k in d.keys():
            text = text.replace(k, d[k])
        return text

    def normalize_en_symbol(self, text):
        """
        规整英文符号为特定的几个符号
        :param text:
        :return:normalized text
        """
        d = self.en_symbol_norm_dict
        for k in d.keys():
            text = text.replace(k, d[k])
        return text

    def normalize_multi_symbol(self, text):
        """
        多个符号同时出现时，保留一个
        :param text:
        :return:normalized text
        """
        new_text = ""

        pre_ch = ""
        ss = self.pause_symbols
        for ch in text:
            if ch in ss and pre_ch in ss:
                continue
            new_text += ch
            pre_ch = ch
        return new_text

    def normalize_text_tail(self, text):
        """
        修改结尾为固定英文句号
        :param text:
        :return:normalized text
        """
        tlen = len(text)
        ss = self.pause_symbols

        if tlen == 0:
            return text
        for i in range(tlen):
            if text[tlen - i - 1] not in ss:
                return text[:tlen - i] + "."

    def normalize_sharp(self, text):
        """
        去除多余#号
        :param text:
        :return:
        """
        text = sub(_sharp_pat_1, "# ;", text)
        text = sub(_sharp_pat_2, "# ,", text)
        text = sub(_sharp_pat_3, "# .", text)
        text = sub(_sharp_pat_4, "# ", text)
        return text


# 测试
if __name__ == "__main__":
    test_text = "我家  住在，黄土，，高坡。。。啊！？，这不是一本书<沃尔>，《易筋经》。@#￥%%第三方！水电费、"

    normalizer = SymbolNormalizer()
    norm_text = normalizer.normalize(test_text)

    print(norm_text)
