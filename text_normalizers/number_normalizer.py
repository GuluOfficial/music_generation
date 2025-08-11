# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/3/1
# Description:文本中的数字规范化处理


from re import split
from re import compile
from re import sub

# 一些数字的正则处理项
# 数字匹配
_number_re = compile(r'[\-]?[0-9]+(\.[0-9]+)*')
# 比分
_compare_num_re = compile(r'[\-]?[0-9\.]+\:[\-]?[0-9\.]+(\:[\-]?[0-9\.]+)*')
# 号码
_phonenum_re = compile(r'(编码:|传真:|电话:|号:|号码:|拨|编码|呼叫|拨打|传真号|传真|电话|号|号码|号码是|电话是|号是|打|致电|代码)([0-9\-]+)')
# 温度
_temp_re = compile(r'([\-]?[0-9]+)(摄氏度|度)')
# 百分比
_hundred_rate_re = compile(r'([\-]?[0-9]+(\.[0-9]+)*)(\%)')

# 年不好确定，只能保证多数情况下正确
_count_year_re1 = compile(r'(前|后|过|过了|活|借|有|命是|死了|活了|存在了|生存了|持续了|持续|维持了|延续了|延续|保存了|保存|保质)([0-9]{4})')
_count_year_re2 = compile(r'([0-9]{4})(年前|年后|年寿|年以前|年以后|年之前|年之后|过后)')
_year_re = compile(r'([0-9]{4})(年)')


class NumberNormalizer:
    """
    文本数字规范化处理，数字到汉字的转换
    """
    _cn_num_dict = {
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九',
        '0': '零',
        '.': '点',
        '+': '正',
        '-': '负',
        ':': '比'
    }

    _special_num_dict = {
        "Ⅰ": "一",
        "Ⅱ": "二",
        "Ⅲ": "三",
        "Ⅳ": "四",
        "Ⅴ": "五",
        "Ⅵ": "六",
        "Ⅶ": "七",
        "Ⅷ": "八",
        "Ⅸ": "九",
        "火警119": "火警幺幺九", "美国911": "美国九幺幺", "911事件": "九幺幺事件", "120急救": "幺二零急救",
        "123木头人": "一二三木头人", "代号47": "代号四十七", "77年航空港": "七七年航空港", "火箭少女101": "火箭少女一零一",
        "京东618": "京东六幺八", "双11": "双十一", "98k": "九八k", "98K": "九八k", "315晚会": "三幺五晚会",
        "360浏览器": "三六零浏览器", "360安全卫士": "三六零安全卫士", "360安全浏览器": "三六零安全浏览器",
        "OPPO R17": "OPPO R十七", "50 Cent": "五十Cent", "创造101": "创造幺零幺",
        "歌曲9420": "歌曲九四二零", "123小红军": "一二三小红军", "1234喜欢": "一二三四喜欢", "英伦86": "英伦八六",
        "985工程": "九八五工程", "985大学": "九八五大学", "211工程": "二幺幺工程", "211大学": "二幺幺大学",
        "2011计划": "二零幺幺计划", '87版': '八七版', '2019尖锋之夜': '二零一九尖锋之夜', '8486网': '八四八六网',
        '黑太阳731': '黑太阳七三幺', '2018中国好声音': '二零一八中国好声音', '58同城': '五八同城',
    }

    def __init__(self):
        pass

    def special_num_to_word(self, text):
        """
        特殊数字转换
        :param text:
        :return:
        """
        d = self._special_num_dict
        for k in d.keys():
            text = text.replace(k, d[k])
        return text

    def base_num_to_word(self, num_str):
        """
        基本的数字到汉字（不带进制）
        :param num_str:
        :return:
        """
        num_dict = self._cn_num_dict
        return "".join([num_dict[num] for num in num_str])

    def float_to_word(self, m=None, num_str=None):
        """
        浮点数到汉字
        :param m:
        :param num_str:
        :return:
        """
        if num_str is None:
            num_str = str(m.group(0))

        prefix = ""
        num_dict = self._cn_num_dict

        if num_str.startswith("-"):
            prefix = num_str["-"]
        num_str = num_str.replace("-", "")

        nums = num_str.split(".")
        if len(num_str) > 20 or len(nums) > 2 or len(nums[0]) > 15:
            return self.base_num_to_word(num_str)

        if len(nums) > 1:
            result = self.integer_to_word(nums[0]) + num_dict['.'] + self.base_num_to_word(nums[1])
        else:
            result = self.integer_to_word(nums[0])

        return prefix + result

    def integer_to_word(self, num_str):
        """
        正整数转换
        :param num_str:
        :return:
        """
        num_dict = self._cn_num_dict
        result = ""

        len_num = len(num_str)
        if len_num == 1:
            return num_dict[num_str[0]]

        _index_name = ['', '十', '百', '千', '万', '十', '百', '千', '亿', '十', '百', '千', '万', '十', '百', '千']
        for i in range(len_num):
            if num_str[i] == '0':
                if len_num - i - 1 == 4:
                    if result[-1] == '零':
                        result = result[0:-1]
                    if result[-1] != '亿':
                        result += '万'
                elif len_num - i - 1 == 8:
                    if result[-1] == '零':
                        result = result[0:-1]
                    result += '亿'
                elif len(result) > 0 and result[-1] != '零':
                    result += '零'
            else:
                result += num_dict[num_str[i]] + _index_name[len_num - i - 1]

        len_result = len(result)
        if result.endswith('零') and len_result > 1:
            result = result[0:-1]
        if result.startswith("一十"):
            result = result[1:]

        return result

    def phone_num_to_word(self, m):
        """
        号码转换
        :param m:
        :return:
        """
        num_dict = self._cn_num_dict.copy()
        num_dict["1"] = "幺"

        num_str = str(m.group(2))
        num_str = num_str.replace("-", "")
        return str(m.group(1)) + "".join([num_dict[num] for num in num_str])

    def temp_num_to_word(self, m):
        """
        温度转换
        :param m:
        :return:
        """
        num_str = m.group(1)
        return self.float_to_word(num_str=num_str).replace("负", "零下") + str(m.group(2))

    def compare_num_to_word(self, m):
        """
        气温数字转换
        :param m:
        :return:
        """
        num_str = str(m.group(0))
        nums = split(u'(:)', num_str)
        num_dict = self._cn_num_dict
        result = ""

        for chs in nums:
            if chs == ":":
                result += num_dict[chs]
            else:
                result += self.float_to_word(num_str=chs)
        return result

    def count_year_num_to_word(self, m):
        """
        计数年转换
        :param m:
        :return:
        """
        text = m.group(0)
        text = sub(_number_re, self.float_to_word, text)
        return text

    def year_num_to_word(self, m):
        """
        普通年份转换
        :param m:
        :return:
        """
        num_str = m.group(1)
        return self.base_num_to_word(num_str) + str(m.group(2))

    def percentage_to_word(self, m):
        """
        百分比转换成汉字
        :param m:
        :return:
        """
        text = m.group(0)
        text = sub(_number_re, self.float_to_word, text)
        return (r'百分之' + text).replace("%", "")

    def normalize(self, text):
        """
        数字规范化处理
        :param text:
        :return:
        """
        text = self.special_num_to_word(text)
        text = sub(_compare_num_re, self.compare_num_to_word, text)
        text = sub(_phonenum_re, self.phone_num_to_word, text)
        text = sub(_temp_re, self.temp_num_to_word, text)
        text = sub(_count_year_re1, self.count_year_num_to_word, text)
        text = sub(_count_year_re2, self.count_year_num_to_word, text)
        text = sub(_year_re, self.year_num_to_word, text)
        text = sub(_hundred_rate_re, self.percentage_to_word, text)
        text = sub(_number_re, self.float_to_word, text)

        return text


# 测试
if __name__ == "__main__":
    test_text = "美国911事件，110我拨打的是电话110，手机号码1871098，比分11:34，现在123.3%是2019年,20年前公元709年他活了109年现在110岁"

    normalizer = NumberNormalizer()
    norm_text = normalizer.normalize(test_text)

    print(norm_text)
