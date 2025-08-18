import os
# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/08/01
# Description:

from datetime import timedelta
from flask import Flask
# 解决终端调用跨域问题
from flask_cors import *

from utils import get_config
from utils import Logger

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "..", "syn_outputs"), static_url_path="/syn_outputs")
@app.route("/healthz")
def healthz():
    return {"ok": True}, 200

app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"
# 解决终端调用跨域问题
CORS(app, supports_credentials=True)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
config = get_config("./server_configs/config_music_generation.yaml")
logger = Logger(config.log_dir, "logger_music_generation.log")
