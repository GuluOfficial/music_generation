# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/8/2
# Description:
import json
from traceback import print_exc

from baiduspider import BaiduSpider


class BaiduScrapy(object):
    def __init__(self):
        self.spider = BaiduSpider()

    def query(self, query_text):
        responses = []
        try:
            results = self.spider.search_zhidao(query=query_text).plain
            for item in results:
                if 'answer' in item and len(item['answer']) > 1:
                    responses.append(str(item['answer'].replace(" ", "")))
                    if len(results) > 10:
                        break
        except Exception:
            print("搜索出错！")
            print_exc()

        code = 1 if len(responses) > 1 else 0
        message = "success" if len(responses) > 1 else "搜索出错！"
        return code, message, responses


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
        print_exc()
        return ip, data
