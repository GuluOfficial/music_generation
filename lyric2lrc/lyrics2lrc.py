#!/usr/bin/env python3
"""
lyrics2lrc.py
Convert a plain-text lyrics file (UTF-8, one line per lyric) into a strict .lrc file.

Strict LRC rules here:
- One timestamp per line, format: [mm:ss.cc] with dot "." as decimal separator.
- mm and ss are zero-padded to 2 digits, cc is centiseconds (00-99).
- Timestamps strictly increase (if equal after rounding, nudged by 0.01s).
- Output is UTF-8 without BOM.
- Optional header can be suppressed with --no-header

Duration heuristics (tunable):
  dur = clamp(min_dur, base + per_char * N + punc_bonus, max_dur)
  where N is the count of non-space, non-punctuation characters.
  A short inter-line gap is added between lines (gap_s).

Usage:
  python lyrics2lrc.py input.txt output.lrc [--no-header] [--base 0.8] [--per-char 0.22]
                                           [--min 1.6] [--max 6.0] [--gap 0.25]
                                           [--seed 42]
"""

import argparse
import math
import random
import re
from pathlib import Path

CN_PUNCT = "，。！？、；：“”‘’（）《》—…·"
ASCII_PUNCT = r""",.!?;:'"()[]{}-–—…"""
ALL_PUNCT = set(CN_PUNCT + ASCII_PUNCT)

def count_core_chars(s: str) -> int:
    # count characters except spaces and punctuation
    return sum(1 for ch in s if (not ch.isspace()) and (ch not in ALL_PUNCT))

def punc_bonus_for_line(s: str) -> float:
    s = s.strip()
    if not s:
        return 0.0
    last = s[-1]
    if last in "，、,;；：":
        return 0.25
    if last in "。！？!?…":
        return 0.40
    return 0.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def sec_to_tag(t: float) -> str:
    # Round to centiseconds
    t = max(0.0, t)
    total_cs = int(round(t * 100.0))
    mm, cs_rem = divmod(total_cs, 6000)  # 6000 cs per minute
    ss, cs = divmod(cs_rem, 100)
    return f"[{mm:02d}:{ss:02d}.{cs:02d}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", type=Path, help="UTF-8 text, one lyric line per line")
    ap.add_argument("output_lrc", type=Path, help="Output .lrc path")
    ap.add_argument("--no-header", action="store_true", help="Do not write ti/ar/by header lines")
    ap.add_argument("--title", type=str, default="Generated", help="LRC [ti:]")
    ap.add_argument("--artist", type=str, default="LocalLM", help="LRC [ar:]")
    ap.add_argument("--base", type=float, default=0.8, help="Base duration per line (s)")
    ap.add_argument("--per-char", type=float, default=0.22, dest="per_char", help="Seconds per non-punct char")
    ap.add_argument("--min", type=float, default=1.6, dest="min_dur", help="Min duration per line (s)")
    ap.add_argument("--max", type=float, default=6.0, dest="max_dur", help="Max duration per line (s)")
    ap.add_argument("--gap", type=float, default=0.25, dest="gap_s", help="Inter-line gap (s)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (if later we add jitter)")
    args = ap.parse_args()

    random.seed(args.seed)

    lines = [ln.rstrip("\n") for ln in args.input_txt.read_text(encoding="utf-8").splitlines()]
    # filter and keep only non-empty lines (you can tweak this if empty lines are meaningful)
    lines = [ln for ln in lines if ln.strip() != ""]

    # compute durations
    durations = []
    for ln in lines:
        N = count_core_chars(ln)
        pb = punc_bonus_for_line(ln)
        dur = args.base + args.per_char * N + pb
        dur = clamp(dur, args.min_dur, args.max_dur)
        durations.append(dur)

    # accumulate start times
    starts = []
    t = 0.0
    for i, d in enumerate(durations):
        # ensure strictly increasing tags after rounding
        if i > 0:
            # nudge time if rounding would collide with previous
            prev_cs = int(round(starts[-1] * 100.0))
            cur_cs = int(round(t * 100.0))
            if cur_cs <= prev_cs:
                t = (prev_cs + 1) / 100.0
        starts.append(t)
        t += d + args.gap_s

    # write LRC
    with args.output_lrc.open("w", encoding="utf-8", newline="\n") as f:
        if not args.no_header:
            f.write(f"[ti:{args.title}]\n")
            f.write(f"[ar:{args.artist}]\n")
            f.write("[by:lyrics2lrc]\n")
            f.write("[offset:0]\n")
        for st, ln in zip(starts, lines):
            tag = sec_to_tag(st)
            f.write(f"{tag} {ln}\n")

    print(f"Wrote {len(lines)} lines to: {args.output_lrc}")

if __name__ == "__main__":
    main()
