# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/9/27
# Description: 音乐生成接口

from flask import request, jsonify
import time
import traceback
import base64

from .app import *
from utils.utils import parse_query_parameters
from pipline_inference_melody_model_v2 import syn_music
from Lyric_generator_t5.generator import generate_lyric, setup_lyric

lyric_model = setup_lyric() # 后续改成class初始化

@app.route('/synthesis_music', methods=['POST'])
def synthesis_music():
    ip, data = parse_query_parameters(request, logger=logger)
    ip_str = ip.replace(".", "_")
    logger.log("用户：%s，发起音乐文本生成请求！" % ip_str)
    
    t_start = time.time()
    if "text" not in data or len(str(data["text"]).strip()) < 1:
        if "keyword" not in data or len(str(data["keyword"]).strip()) < 1:
            json_result = {"code": 201, "message": "输入参数错误，必须输入非空字段text！"}
            return jsonify(json_result)
        else:
            text = data["keyword"]
            try:
                if text == "宁夏" or text == "青春修炼手册":
                    wav_path = f"./resources/{text}.wav"
                    text_path = f"./resources/{text}.txt"
                    with open(wav_path, "rb") as f:
                        audio_str = f.read()
                    bsstr = base64.b64encode(audio_str)
                    bin_str = bsstr.decode()

                    with open(text_path, "rt", encoding="utf-8") as f:
                        text = ""
                        for l in f:
                            text = text + l.strip() + "。"

                    json_result = {
                            "code": 200,
                            "message": "success！",
                            "data":
                            {
                                "lyrics": text,
                                "audio":bin_str,
                            }
                    }
                    return jsonify(json_result)
            except Exception:
                traceback.print_exc()
                print("预设内容出错!")

            text = generate_lyric(text, lyric_model)
            # text = '。'.join(text.split('。')[:8])
    else:
        text = data["text"]
    print(text)
    print(len(text))
    if len(text) < 1 or len(text) > 500:
        json_result = {"code": 202, "message": "生成失败，传入文本必须大于1个字符，小于50000个字符！"}
        return jsonify(json_result)
    
    # Test
    #json_result = {
    #    "code": 200,
    #    "message": "success！",
    #    "data":
    #        {
    #            "lyrics": text,
    #            "audio":None,
    #        }
    #}
    #logger.log(f"音频合成耗时：{time.time() - t_start} 秒。")
    #return jsonify(json_result)

    try:
        t_start = time.time()
        wav_data = syn_music(text)
        bin_str = base64.b64encode(wav_data).decode()
        # audio_str = "data:audio/ogg;base64,%s" % bin_str
        traceback.print_exc()
        json_result = {
            "code": 200,
            "message": "success！",
            "data":
                {
                    "lyrics": text,
                    "audio":bin_str,
                }
        }
        logger.log(f"音频合成耗时：{time.time() - t_start} 秒。")
        return jsonify(json_result)
    except:
        traceback.print_exc()
        json_result = {"code": 203, "message": "服务器错误，歌声合成出错！"}
        return jsonify(json_result)
