#!/usr/bin/env python3
"""
lrc_validate_normalize.py
Validate and normalize an LRC file to a strict form acceptable by some picky consumers.

- Accepts:
    [mm:ss] or [mm:ss.cc] or [mm:ss,cc] (comma gets normalized to dot)
    multiple timestamps per line like [00:10.00][00:20.00] lyric

- Produces:
    * UTF-8 (no BOM)
    * One timestamp per line
    * [mm:ss.cc] with dot, two centiseconds digits
    * Lines sorted by time, strictly increasing by at least 0.01s
    * Optional header preserved (ti/ar/by/offset), rewritten first

Usage:
  python lrc_validate_normalize.py input.lrc output.lrc
"""

import re
import sys
from pathlib import Path

TIME_RE = re.compile(
    r"""\[
        (?P<mm>\d{1,3})
        :
        (?P<ss>\d{2})
        (?:
            [\.:,]
            (?P<frac>\d{1,3})
        )?
    \]""",
    re.VERBOSE,
)

HEADER_RE = re.compile(r"^\[(ti|ar|al|by|offset):.*\]$", re.IGNORECASE)

def to_seconds(mm, ss, frac):
    m = int(mm)
    s = int(ss)
    f = frac or "0"
    # normalize to centiseconds
    if len(f) > 2:
        # round to 2 digits
        v = round(int(f) / (10 ** (len(f) - 2)))
    else:
        v = int(f.ljust(2, "0"))
    return m * 60 + s + v / 100.0

def sec_to_tag(t):
    if t < 0: t = 0.0
    total_cs = int(round(t * 100.0))
    mm, cs_rem = divmod(total_cs, 6000)
    ss, cs = divmod(cs_rem, 100)
    return f"[{mm:02d}:{ss:02d}.{cs:02d}]"

def main():
    if len(sys.argv) < 3:
        print("Usage: python lrc_validate_normalize.py input.lrc output.lrc")
        sys.exit(1)
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])

    raw = inp.read_bytes()
    # detect UTF-8 BOM
    bom = raw.startswith(b"\xEF\xBB\xBF")
    text = raw.decode("utf-8-sig")  # strips BOM if present

    headers = []
    entries = []  # list of (seconds, lyric)

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue

        if HEADER_RE.match(line):
            headers.append(line)
            continue

        times = TIME_RE.findall(line)
        lyric = TIME_RE.sub("", line).strip()
        if not times:
            # line without time tag: skip but report
            print(f"WARNING: no timestamp found, skipping line: {line}")
            continue

        for (mm, ss, frac) in times:
            ts = to_seconds(mm, ss, frac)
            entries.append((ts, lyric))

    # sort by time and enforce strictly increasing
    entries.sort(key=lambda x: x[0])
    fixed = []
    prev_cs = -1
    for ts, lyric in entries:
        cs = int(round(ts * 100.0))
        if cs <= prev_cs:
            cs = prev_cs + 1
        prev_cs = cs
        fixed.append((cs / 100.0, lyric))

    # Write out
    with outp.open("w", encoding="utf-8", newline="\n") as f:
        # Keep only these headers in canonical order if present
        order = ["ti", "ar", "al", "by", "offset"]
        hdr_map = {}
        for h in headers:
            k = h[1:h.find(":")].lower()
            hdr_map[k] = h
        for k in order:
            if k in hdr_map:
                f.write(hdr_map[k] + "\n")
        for ts, lyric in fixed:
            f.write(f"{sec_to_tag(ts)} {lyric}\n")

    print(f"Normalized LRC written to: {outp} (lines: {len(fixed)})")
    if bom:
        print("NOTE: Input had UTF-8 BOM; output is UTF-8 without BOM.")
    # basic stats
    if fixed:
        total = fixed[-1][0]
        print(f"Duration (last timestamp): {total:.2f}s")

if __name__ == "__main__":
    main()
