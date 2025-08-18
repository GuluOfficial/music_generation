#!/usr/bin/env bash
set -euo pipefail

# Orchestrator: request.json(string or path) -> lyrics.txt -> rough.lrc -> final.lrc -> song.wav
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <request.json string or path>" >&2
  exit 1
fi

REQ_ARG="$1"

REQ_ROOT="$ROOT/data/requests"
mkdir -p "$REQ_ROOT"

# Compute next request id
NEXT_ID="$(python3 - <<'PY' "$REQ_ROOT"
import os, re, pathlib, sys
p=pathlib.Path(sys.argv[1])
p.mkdir(parents=True, exist_ok=True)
ids=[]
for d in p.iterdir():
    if d.is_dir():
        m=re.match(r'REQ-(\d{5})$', d.name)
        if m: ids.append(int(m.group(1)))
n=(max(ids)+1 if ids else 1)
print(f"REQ-{n:05d}")
PY
)"

RUN_DIR="$REQ_ROOT/$NEXT_ID"
mkdir -p "$RUN_DIR"

# Save request.json
if [[ -f "$REQ_ARG" ]]; then
  cp "$REQ_ARG" "$RUN_DIR/request.json"
else
  printf "%s\n" "$REQ_ARG" > "$RUN_DIR/request.json"
fi

# 1) Generate lyrics (Ollama). Writes to lyrics.txt
"$ROOT/bin/gen_lyrics_ollama.sh" "$RUN_DIR/request.json" > "$RUN_DIR/lyrics.txt"

# 2) Heuristic timestamps -> rough.lrc (no header)
python3 "$ROOT/py/lyrics2lrc.py" "$RUN_DIR/lyrics.txt" "$RUN_DIR/rough.lrc" --no-header

# 3) Normalize -> final.lrc
python3 "$ROOT/py/lrc_validate_normalize.py" "$RUN_DIR/rough.lrc" "$RUN_DIR/final.lrc"

# 4) Generate a per-request DiffRhythm runner and execute it
GEN="$ROOT/bin/gen_diffrhythm_runner.sh"
RUNNER="$("$GEN" "$RUN_DIR")"
bash "$RUNNER"

# 5) Update latest pointer or copy (if symlinks not allowed)
if ln -sfn "$RUN_DIR" "$ROOT/data/latest" 2>/dev/null; then
  :
else
  echo "Symlink not permitted, copying artifacts into data/latest"
  rm -rf "$ROOT/data/latest"
  mkdir -p "$ROOT/data/latest"
  cp -f "$RUN_DIR/lyrics.txt" "$ROOT/data/latest/" 2>/dev/null || true
  cp -f "$RUN_DIR/final.lrc"  "$ROOT/data/latest/" 2>/dev/null || true
  cp -f "$RUN_DIR/song.wav"   "$ROOT/data/latest/" 2>/dev/null || true
  echo "$RUN_DIR" > "$ROOT/data/LATEST_PATH.txt"
fi

# 6) Publish to syn_outputs/ (for UI)
PUB="$ROOT/../syn_outputs/$NEXT_ID"
mkdir -p "$PUB"
cp -f "$RUN_DIR/lyrics.txt" "$PUB/"
cp -f "$RUN_DIR/final.lrc"  "$PUB/"
cp -f "$RUN_DIR/song.wav"   "$PUB/"
mkdir -p "$ROOT/../syn_outputs/latest"
cp -f "$RUN_DIR/"{lyrics.txt,final.lrc,song.wav} "$ROOT/../syn_outputs/latest/" 2>/dev/null || true

echo "request_id=$NEXT_ID"
echo "lyrics_txt=$RUN_DIR/lyrics.txt"
echo "final_lrc=$RUN_DIR/final.lrc"
echo "audio_path=$RUN_DIR/song.wav"

