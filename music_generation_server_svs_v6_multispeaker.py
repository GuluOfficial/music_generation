# �?- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/12/23
# Description: v3

import os

from controllers_music_generation_svs_v6_multispeaker import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
