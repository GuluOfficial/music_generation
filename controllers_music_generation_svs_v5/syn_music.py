# —*- coding: utf-8 -*-
# Author: zsk
# Creator Date: 2022/9/27
# Modified: 2025/8/12
# Description: 音乐生成接口
import os, json, subprocess, base64  # base64 already imported in your file
from flask import request, jsonify
import time
import traceback
import base64

from .app import *
from utils.utils import parse_query_parameters
from pipline_inference_svs_v5 import syn_music
from Lyric_generator_t5.generator import generate_lyric, setup_lyric
import sys

env = os.environ.copy()
env.setdefault("DIFFRHYTHM_REPO", os.path.join(_REPO_ROOT, "DiffRhythm"))
env["PY"] = sys.executable                    # <- use THIS Python (drunner)
env["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "6")
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
env["WANDB_DISABLED"] = "true"
env.setdefault("HF_HOME", "/18T/app/hf_cache")

lyric_model = setup_lyric() # 后续改成class初始化

@app.route('/synthesis_music', methods=['POST'])
def synthesis_music():
    ip, data = parse_query_parameters(request, logger=logger)
    ip_str = ip.replace(".", "_")
    logger.log("用户：%s，发起音乐文本生成请求！" % ip_str)
    
    
    if "text" not in data or len(str(data["text"]).strip()) < 1:
        if "keyword" not in data or len(str(data["keyword"]).strip()) < 1:
            json_result = {"code": 201, "message": "输入参数错误，必须输入非空字段text！"}
            return jsonify(json_result)
        else:
            text = data["keyword"]
            logger.log(f"传入关键词:{text}")
            text = generate_lyric(text, lyric_model).replace("-", "")
            #text = '，'.join(text.split('。'))
            logger.log(f"生成歌词：{text}")
    else:
        text = data["text"]
        logger.log(f"直接传入歌词:{text}")

    if len(text) < 1 or len(text) > 500:
        json_result = {"code": 202, "message": "生成失败，传入文本必须大于1个字符，小于50000个字符！"}
        return jsonify(json_result)

        try:
        t_start = time.time()

        # Flip this at runtime: export MUSIC_BACKEND=new|old
        use_new = os.getenv("MUSIC_BACKEND", "new") == "new"

        if not use_new:
            # ===== OLD BACKEND (unchanged) =====
            wav_data = syn_music(text)
            bin_str = base64.b64encode(wav_data).decode()
            json_result = {"code": 200, "message": "success！", "data": {"lyrics": text, "audio": bin_str}}
            logger.log(f"音频合成耗时：{time.time() - t_start} 秒。")
            return jsonify(json_result)

        # ===== NEW BACKEND via music_pipeline =====
        # repo root = parent of this controllers folder
        _APP_DIR = os.path.dirname(os.path.abspath(__file__))
        _REPO_ROOT = os.path.dirname(_APP_DIR)
        pipeline = os.path.join(_REPO_ROOT, "music_pipeline", "bin", "pipeline.sh")

        # Build a request for the pipeline; pass raw lyrics so we skip Ollama here
        req_dict = {
            "raw_lyrics":  text,
            "ref_prompt":  data.get("ref_prompt", "hopeful,ballad,pop,emotion"),
            "audio_length": int(data.get("audio_length", 95)),
            "style":       data.get("style", "现代流行"),
            "lines":       int(data.get("lines", 6)),
        }
        req_str = json.dumps(req_dict, ensure_ascii=False)

        env = os.environ.copy()
        # Point to DiffRhythm repo if not provided
        if "DIFFRHYTHM_REPO" not in env:
            candidate = os.path.join(_REPO_ROOT, "DiffRhythm")
            if os.path.isdir(candidate):
                env["DIFFRHYTHM_REPO"] = candidate
        # (optional) auto-activate conda env for DiffRhythm if you use one
        env.setdefault("DIFFRHYTHM_CONDA_ENV", "diffrhythm")

        proc = subprocess.run(
            ["bash", pipeline, req_str],
            cwd=os.path.join(_REPO_ROOT, "music_pipeline"),
            env=env, capture_output=True, text=True
        )
        if proc.returncode != 0:
            logger.log(f"pipeline error: {proc.stderr or proc.stdout}")
            # graceful fallback to old path
            wav_data = syn_music(text)
            bin_str = base64.b64encode(wav_data).decode()
            json_result = {"code": 200, "message": "success！", "data": {"lyrics": text, "audio": bin_str}}
            logger.log(f"[fallback] 音频合成耗时：{time.time() - t_start} 秒。")
            return jsonify(json_result)

        # parse key=value lines from pipeline stdout
        info = {}
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()

        audio_path = info.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            logger.log("pipeline ok but audio_path missing; falling back")
            wav_data = syn_music(text)
            bin_str = base64.b64encode(wav_data).decode()
            json_result = {"code": 200, "message": "success！", "data": {"lyrics": text, "audio": bin_str}}
            logger.log(f"[fallback] 音频合成耗时：{time.time() - t_start} 秒。")
            return jsonify(json_result)

        with open(audio_path, "rb") as f:
            wav_bytes = f.read()
        bin_str = base64.b64encode(wav_bytes).decode()

        json_result = {
            "code": 200,
            "message": "success！",
            "data": {"lyrics": text, "audio": bin_str}
        }
        logger.log(f"音频合成耗时：{time.time() - t_start} 秒。")
        return jsonify(json_result)

    except Exception:
        traceback.print_exc()
        json_result = {"code": 203, "message": "服务器错误，歌声合成出错！"}
        return jsonify(json_result)
