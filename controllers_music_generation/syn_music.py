# -*- coding: utf-8 -*-
# Description: 音乐生成接口 (pipeline-only)
import os, sys, json, subprocess, base64, time, traceback
from flask import request, jsonify
from pathlib import Path
from .app import *
from utils.utils import parse_query_parameters

import re  # add if missing
import re

LEGACY_PHRASES = ["在夜色里慢慢发光", "沿着河岸把心事流淌", "风把故事吹成了月光", "唱进你的心房"]

def looks_like_topic(s: str) -> bool:
    if not s: 
        return False
    if "\n" in s:
        return False
    if any(p in s for p in "。！？!?，、；"):
        return False
    return len(s) <= 16

def detect_legacy_template(text: str) -> str | None:
    """
    Return the theme if the classic 4-line template is detected, else None.
    Examples that should match:
      玫瑰在夜色里慢慢发光
      我把玫瑰唱进你的心房
    """
    if not text:
        return None
    found = 0
    for key in LEGACY_PHRASES:
        if key in text:
            found += 1
    if found < 2:
        return None  # probably not the legacy template

    # try to extract the noun/theme
    # 1) “我把{X}唱进你的心房”
    m = re.search(r"我把([^\n，。！？!?、；]{1,20})唱进你的心房", text)
    if m:
        theme = m.group(1).strip()
        if theme:
            return theme

    # 2) “{X}在夜色里慢慢发光”
    m = re.search(r"([^\n，。！？!?、；]{1,20})在夜色里慢慢发光", text)
    if m:
        theme = m.group(1).strip()
        if theme:
            return theme

    return None

def normalize_to_lines(s: str) -> str:
    lines = [seg.strip() for seg in re.split(r'[。\n]+', s) if seg.strip()]
    return "\n".join(lines)

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

    if not theme:
        m = re.search(r"我把(.+?)唱进你的心房", lines[-1])
        if m:
            theme = m.group(1).strip("，。！？、；:：  ")

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
    # Decide mode (strict precedence: keyword/theme > legacy-template > raw)
    payload = {
        "audio_length": audio_len,
        "ref_prompt": ref_prompt,
    }

    # 0) If keyword exists, ALWAYS use theme generation and ignore text/raw_lyrics
    if kw:
        payload["keyword"] = kw
        mode = "theme_from_keyword"

    else:
        # Check provided fields
        raw  = str(data.get("raw_lyrics", "")).strip()
        text = txt  # from earlier

        # 1) Legacy 4-line template sneaking in via raw or text? Extract theme.
        theme_from_legacy = detect_legacy_template(raw or text)
        if theme_from_legacy:
            payload["keyword"] = theme_from_legacy
            mode = "theme_from_legacy_template"

        # 2) Short, unpunctuated 'text' like “玫瑰” → treat as theme
        elif text and looks_like_topic(text):
            payload["keyword"] = text
            mode = "theme_from_text"

        # 3) True raw lyrics (multi-line user lyrics) → allow raw
        elif raw:
            payload["raw_lyrics"] = normalize_to_lines(raw)
            mode = "raw_lyrics"

        elif text:
            # Multi-line or punctuated text presumed to be actual lyrics
            payload["raw_lyrics"] = normalize_to_lines(text)
            mode = "raw_from_text"

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
