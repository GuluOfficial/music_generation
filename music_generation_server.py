# -*- coding: utf-8 -*-
from controllers_music_generation import *  # provides `app`
# near top of music_generation_server.py
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")
def start_server():
    from controllers_music_generation.app import app  # ensure the same app
    app.run(host="0.0.0.0", port=5005, debug=True, threaded=True, use_reloader=False)

if __name__ == "__main__":
    start_server()
