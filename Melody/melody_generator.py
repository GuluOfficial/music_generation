# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/9/27
# Email: skzheng@163.com
# Description:

import torch
import logging
import traceback

from Melody.telemelody.model import Lyric2Melody
from Melody.telemelody.utils import process


logging.basicConfig(filename=__name__ + '.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

seed = 9
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
lyric2melody = Lyric2Melody()


# 输入歌词，获取旋律
def get_melody(lyrics):
    try:
        lyrics = process(lyrics)
        midi_obj, durations, pitches = lyric2melody.generate_melody(lyrics)
        return midi_obj, lyrics
    except:
        logging.ERROR("旋律生成出错")
        traceback.print_exc()
        return None
