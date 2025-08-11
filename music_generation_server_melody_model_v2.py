# �?- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/8/30
# Description: v2

import os

from controllers_music_generation_melody_v2 import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def start_server():
    """
    ??
    """
    host = config.host
    port = config.port
    app.run(host=host, port=int(port), debug=True, threaded=True, use_reloader=False)
    #app.run(host=host, port=int(port), debug=True, threaded=True, use_reloader=False, ssl_context=(os.path.join(config.cert_file_path, 'server.crt'), os.path.join(config.cert_file_path, 'server.key')))


if __name__ == "__main__":
    start_server()
