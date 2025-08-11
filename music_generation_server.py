# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/8/30
# Description: 音乐生成服务

import os

from controllers_music_generation import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def start_server():
    """
    启动服务
    """
    host = config.host
    port = config.port
    app.run(host=host, port=int(port), debug=True, threaded=True, use_reloader=False,
            ssl_context=(os.path.join(config.cert_file_path, 'server.crt'), os.path.join(config.cert_file_path, 'server.key')))


if __name__ == "__main__":
    start_server()
