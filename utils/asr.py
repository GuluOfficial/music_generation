# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/11/18
# Description:

import json
import sys
import time
from traceback import print_exc
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request
from urllib.request import urlopen

timer = time.perf_counter

if sys.platform == "win32":
    timer = time.clock
else:
    timer = time.time

API_KEY_BAIDU = 'W5bNWgrba7f7Nre1wWDGOA33'
SECRET_KEY_BAIDU = 'rJaHMBmhHLREaRg21oLkVPAhcCtrpAD9'
DEV_PID_BAIDU = 1537
ASR_URL_BAIDU = 'http://vop.baidu.com/server_api'
SCOPE_BAIDU = 'audio_voice_assistant_get'
TOKEN_URL_BAIDU = 'http://openapi.baidu.com/oauth/2.0/token'
CUID_BAIDU = 'baidu_workshop'
IS_PY3 = True

# 采样率
SAMPLERATE = 16000


class DemoError(Exception):
    pass


def fetch_token(logger):
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY_BAIDU,
              'client_secret': SECRET_KEY_BAIDU}
    post_data = urlencode(params)
    if (IS_PY3):
        post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL_BAIDU, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        # print('token http response http code : ' + str(err.code))
        result_str = err.read()
    if (IS_PY3):
        result_str = result_str.decode()

    # print(result_str)
    result = json.loads(result_str)
    # print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if SCOPE_BAIDU and (not SCOPE_BAIDU in result['scope'].split(' ')):  # SCOPE = False 忽略检查
            raise DemoError('scope is not correct')
        # print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')


"""  TOKEN end """


def asr_baidu(audio_file, logger):
    token = fetch_token(None)

    """
    httpHandler = urllib2.HTTPHandler(debuglevel=1)
    opener = urllib2.build_opener(httpHandler)
    urllib2.install_opener(opener)
    """
    FORMAT = audio_file[-3:]
    speech_data = []
    with open(audio_file, 'rb') as speech_file:
        speech_data = speech_file.read()
    length = len(speech_data)
    if length == 0:
        raise DemoError('file %s length read 0 bytes' % audio_file)

    params = {'cuid': CUID_BAIDU, 'token': token, 'dev_pid': DEV_PID_BAIDU}
    # 测试自训练平台需要打开以下信息
    # params = {'cuid': CUID, 'token': token, 'dev_pid': DEV_PID, 'lm_id' : LM_ID}
    params_query = urlencode(params);

    headers = {
        'Content-Type': 'audio/' + FORMAT + '; rate=' + str(SAMPLERATE),
        'Content-Length': length
    }

    url = ASR_URL_BAIDU + "?" + params_query
    req = Request(ASR_URL_BAIDU + "?" + params_query, speech_data, headers)
    try:
        begin = timer()
        f = urlopen(req)
        result_str = f.read()
        print("baidu response:", result_str)
        # print("Request time cost %f" % (timer() - begin))
    except URLError as err:
        # print('asr http response http code : ' + str(err.code))
        print_exc()
        result_str = err.read()

    if (IS_PY3):
        result_str = str(result_str, 'utf-8')
    # print(result_str)

    return json.loads(result_str)["result"][0]


if __name__ == "__main__":
    audio_file = r"C:\Users\DELL\AppData\Local\Temp\tmpa4g81n6j._16k.wav"
    save_file = r"D:/result.txt"
    results = []
    result = asr_baidu(audio_file, None)
    print(result)
    results.append(result)

    with open(save_file, mode='wt', encoding='utf-8') as f:
        for i, t in enumerate(results):
            f.write(str(i) + "\t" + t + "\n")
