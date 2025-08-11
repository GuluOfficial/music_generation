# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2021/11/17
# Description: 解析请求

import json
import traceback
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import tempfile


def normalize_query_text(text):
    """
    预处理输入文本
    :param text:
    :return:
    """
    new_text = text.replace("#", "")
    return new_text


def get_time_str():
    """
    获取时间字符串
    :return:
    """
    date_format = '%Y-%m-%d %H:%M:%S.%f'
    date_str = datetime.now().strftime(date_format)
    return date_str


def parse_query_parameters(request, logger):
    """
    从请求中获取参数
    :return:
    """
    data = {}
    ip = request.remote_addr
    try:
        if request.content_type.startswith('application/json'):
            data = request.get_data()
            data = json.loads(data)
        else:
            for key, value in request.form.items():
                if key.endswith('[]'):
                    data[key[:-2]] = request.form.getlist(key)
                else:
                    data[key] = value
        logger.log("终端访问地址：" + str(ip))
        return ip, data
    except:
        traceback.print_exc()
        return ip, data


def get_config(config_file_path, logger=None):
    """
    获取配置文件信息
    :param config_file_path:
    :param logger:
    :return:
    """
    try:
        with open(config_file_path, 'rt', encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.Loader)
            config = edict(config)
            return config
    except Exception as e:
        traceback.print_exc()
        if logger is not None:
            #logger.log("Load config file failure!")
            pass
        else:
            print("Load config file failure!")
        exit(1)


def get_session_uuid(ip):
    """
    获取对话uuid
    :param ip:
    :return:
    """
    date_str = get_time_str().replace("-", "_").replace(":", "_").replace(".", "_").replace(" ", "_")
    ip_str = ip.replace(".", "_")
    uuid = "ip-"+ip_str + "-t-" + date_str
    return uuid


def get_dialogue_uuid(user_id):
    """
    获取对话uuid
    :param user_id:
    :return:
    """
    date_str = get_time_str().replace("-", "_").replace(":", "_").replace(".", "_")
    date_str = date_str[:date_str.find(" ")]
    uuid = user_id + "-t-" + date_str
    return uuid


def get_utterance_uuid():
    """
    获取对话uuid
    :return:
    """
    date_str = get_time_str().replace("-", "_").replace(":", "_").replace(".", "_").replace(" ", "_")
    uuid = "utt_" + date_str
    return uuid


def write_temp_file(data, suffix, logger, mode='w+b'):
    """
    保存临时文件
    :param data:
    :param suffix:
    :param mode:
    :return:
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, mode=mode, delete=False) as f:
        f.write(data)
        filename = f.name
    logger.log('filename is {}'.format(filename))
    return filename


def singleton(cls):
    """
    构造单例
    :param cls:
    :return:
    """
    instances = {}

    def get_instance(*args, **kwargs):
        """
        获取实例
        :param args:
        :param kwargs:
        :return:
        """
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def padding_b64str(b64str):
    if len(b64str) % 3 == 1:
        b64str += "="
    elif len(b64str) % 3 == 2:
        b64str += "=="
    return b64str