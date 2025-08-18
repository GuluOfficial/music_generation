# -*- coding: utf-8 -*-
# Description: 音乐生成接口 (pipeline-only)
import os, sys, json, subprocess, base64, time, traceback
from flask import request, jsonify
from pathlib import Path
from .app import *
from utils.utils import parse_query_parameters

import re  # add if missing

def extract_theme_from_template(raw: str) -> str | None:
    """
    Detect the legacy 4-line template like:
      1) <THEME>在夜色里慢慢发光
      2) 沿着河岸把心事流淌
      3) 风把故事吹成了月光
      4) 我把<THEME>唱进你的心房
    Return THEME if detected, else None.
    """
    lines = [L.strip() for L in re.split(r'[\r\n]+', raw or '') if L.strip()]
    if len(lines) != 4:
        return None

    # Try to get theme from line 1 by stripping known suffixes
    suffixes = [
        "在夜色里慢慢发光",
        "在夜色里悄悄发光",
        "在黑夜里慢慢发光",
        "在夜色里缓缓发光",
    ]
    theme = None
    for suf in suffixes:
        if lines[0].endswith(suf) and len(lines[0]) > len(suf):
            theme = lines[0][:-len(suf)].strip("，。！？、；:：  ")
            break

    # Fallback: try from line 4 (“我把<THEME>唱进你的心房”)
    if not theme:
        m = re.search(r"我把(.+?)唱进你的心房", lines[-1])
        if m:
            theme = m.group(1).strip("，。！？、；:：  ")

    # Basic sanity
    if theme and 0 < len(theme) <= 16:
        return theme
    return None


def _keyword_to_lines(kw: str) -> str:
    kw = (kw or "").strip()
    if not kw:
        return ""
    # very light inline lyricer so we don't depend on external models
    lines = [
        f"{kw}在夜色里慢慢发光",
        f"沿着河岸把心事流淌",
        f"风把故事吹成了月光",
        f"我把{kw}唱进你的心房",
    ]
    return "\n".join(lines)

@app.route('/synthesis_music', methods=['POST'])
def synthesis_music():
    ip, data = parse_query_parameters(request, logger=logger)
    ip_str = ip.replace(".", "_")
    logger.log(f"用户：{ip_str}，发起音乐文本生成请求！ 原始data={data}")

    # Pull fields
    txt = str(data.get("text", "") or "").strip()
    kw  = str(data.get("keyword", data.get("theme", "")) or "").strip()
    raw = str(data.get("raw_lyrics", "") or "").strip()
    audio_len  = int(data.get("audio_length", 95) or 95)
    ref_prompt = str(data.get("ref_prompt", "hopeful,ballad,pop,emotion") or "").strip()
    style = str(data.get("style", "") or "").strip()

    # If legacy 4-line template arrives as raw_lyrics, convert it to a theme
    theme_from_template = extract_theme_from_template(raw) if raw else None
    if theme_from_template:
        logger.log(f"检测到四句模板，提取主题：{theme_from_template} -> 走Ollama歌词生成")
        kw = theme_from_template
        raw = ""  # ignore the template content

    # Heuristic: short, unpunctuated 'text' means it's actually a topic
    def looks_like_topic(s: str) -> bool:
        if not s or "\n" in s:
            return False
        if any(p in s for p in "。！？!?，、；"):
            return False
        return len(s) <= 16

    # Build payload for pipeline
    payload = {"audio_length": audio_len, "ref_prompt": ref_prompt}
    if style:
        payload["style"] = style

    # Decide mode
    if kw:
        payload["keyword"] = kw
        mode = "theme"
    elif txt:
        if looks_like_topic(txt):
            payload["keyword"] = txt
            mode = "theme_from_text"
        else:
            # Looks like lyrics -> normalize to one-per-line
            lines = [seg.strip() for seg in re.split(r'[。\n]+', txt) if seg.strip()]
            payload["raw_lyrics"] = "\n".join(lines)
            mode = "raw_from_text"
    elif raw:
        payload["raw_lyrics"] = raw
        mode = "raw"
    else:
        return jsonify({"code": 201, "message": "输入参数错误：请提供 keyword/theme（主题）或 text/raw_lyrics（歌词）。"})

    logger.log(f"判定模式：{mode}；audio_length={audio_len}；ref_prompt={ref_prompt}；payload_keys={list(payload.keys())}")

    # Run pipeline
    try:
        cmd = ["bash", "music_pipeline/bin/pipeline.sh", json.dumps(payload, ensure_ascii=False)]
        logger.log(f"运行管线命令：{cmd}")
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.log("管线输出：\n" + run.stdout.decode("utf-8", errors="ignore"))
    except subprocess.CalledProcessError as e:
        logger.log("管线失败：\n" + e.stdout.decode("utf-8", errors="ignore") + "\nERR:\n" + e.stderr.decode("utf-8", errors="ignore"))
        return jsonify({"code": 203, "message": "服务器错误：生成失败（pipeline）"})

    # Collect newest request dir
    req_root = Path("music_pipeline/data/requests")
    req_dirs = sorted([p for p in req_root.glob("REQ-*") if p.is_dir()])
    if not req_dirs:
        return jsonify({"code": 203, "message": "服务器错误：找不到请求输出目录。"})
    latest = req_dirs[-1]

    # Read outputs
    lrc_path = latest / "final.lrc"
    if not lrc_path.exists():
        lrc_path = latest / "rough.lrc"
    wavs = list((latest / "diffrhythm_output").glob("*.wav"))

    audio_b64 = None
    if wavs:
        with open(wavs[0], "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")

    lyrics_txt = (latest / "lyrics.txt").read_text(encoding="utf-8", errors="ignore") if (latest / "lyrics.txt").exists() else None
    lrc_txt = lrc_path.read_text(encoding="utf-8", errors="ignore") if lrc_path.exists() else None

    return jsonify({
        "code": 200,
        "message": "success！",
        "data": {
            "mode": mode,
            "lyrics": lyrics_txt,
            "lrc": lrc_txt,
            "audio": audio_b64,
        }
    })
