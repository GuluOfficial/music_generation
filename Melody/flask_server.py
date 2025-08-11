import traceback
from flask import Flask, request
import requests
import re
import time
import torch
import logging

from Melody.telemelody.model import Lyric2Melody
from Melody.telemelody.utils import process

logging.basicConfig(filename=__name__ + '.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

seed = 9
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
app = Flask(__name__)

lyric2melody = Lyric2Melody()


@app.route('/')
def server_melody():
    try:
        lyrics = request.args.get('lyrics')
        lyrics = process(lyrics)

        midi_obj, durations, pitchs = lyric2melody.generate_melody(lyrics)

        return {
            "lyrics": lyrics,
            "durations": durations,
            "pitch": pitchs
            # "midi_odj": midi_obj
        }

    except Exception as e:
        traceback.print_exc()
        print(e.args)
        print(str(e))
    return "歌词获取出错！"


app.run(host='0.0.0.0', port=6900, debug=True, threaded=True, use_reloader=False)
