#!/usr/bin/env bash
set -euo pipefail
cd "/home/tiankai/DiffRhythm"
export PYTHONPATH="$PYTHONPATH:$PWD"
export CUDA_VISIBLE_DEVICES=0

# macOS espeak note preserved from your original script
if [[ "${OSTYPE:-}" =~ ^darwin ]]; then
  export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

PY="${PYTHON:-python3}"
$PY infer/infer.py \
  --lrc-path "/media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/final.lrc" \
  --ref-prompt "hopeful,ballad,pop,emotion" \
  --audio-length 95 \
  --repo-id "ASLP-lab/DiffRhythm-1_2" \
  --output-dir "/media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/diffrhythm_output" \
  --chunked \
  --batch-infer-num 5

# Copy newest WAV to the canonical place the pipeline expects
LATEST=$(ls -1t "/media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/diffrhythm_output"/*.wav 2>/dev/null | head -n1 || true)
if [[ -n "$LATEST" ]]; then
  cp -f "$LATEST" "/media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/song.wav"
  echo "Wrote: /media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/song.wav"
else
  echo "ERROR: No WAV found in /media/tiankai/Awesome/lyric2lrc/music_pipeline/data/requests/REQ-00006/diffrhythm_output" >&2
  exit 2
fi
