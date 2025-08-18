#!/usr/bin/env bash
set -euo pipefail
LRC="$1"
OUT="$2"

# Optional: auto-activate conda env for DiffRhythm
if [[ -n "${DIFFRHYTHM_CONDA_ENV:-}" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi
  conda activate "$DIFFRHYTHM_CONDA_ENV"
fi

# If DIFFRHYTHM_CMD is set to an executable, use it
if [[ -n "${DIFFRHYTHM_CMD:-}" && -f "${DIFFRHYTHM_CMD}" && -x "${DIFFRHYTHM_CMD}" ]]; then
  # Expect the cmd to accept --lyrics and --output
  eval "\"$DIFFRHYTHM_CMD\"" --lyrics "$LRC" --output "$OUT"
  exit 0
fi

# If DIFFRHYTHM_CMD is a directory, treat it as the DiffRhythm repo and run infer/infer.py
if [[ -n "${DIFFRHYTHM_CMD:-}" && -d "${DIFFRHYTHM_CMD}" ]]; then
  DIFF_DIR="${DIFFRHYTHM_CMD%/}"
  PY="${PYTHON:-python3}"
  REF_PROMPT="${REF_PROMPT:-hopeful,ballad,pop,emotion}"
  AUDIO_LEN="${AUDIO_LEN:-95}"
  REPO_ID="${REPO_ID:-ASLP-lab/DiffRhythm-1_2}"
  OUT_DIR="${OUT_DIR:-$DIFF_DIR/infer/example/output}"
  mkdir -p "$OUT_DIR"
  pushd "$DIFF_DIR" >/dev/null
  $PY infer/infer.py \
    --lrc-path "$LRC" \
    --ref-prompt "$REF_PROMPT" \
    --audio-length "$AUDIO_LEN" \
    --repo-id "$REPO_ID" \
    --output-dir "$OUT_DIR" \
    ${CHUNKED:+--chunked} \
    ${BATCH_INFER_NUM:+--batch-infer-num "$BATCH_INFER_NUM"}
  popd >/dev/null
  # pick the newest wav from OUT_DIR to copy as $OUT
  LATEST="$(ls -1t "$OUT_DIR"/*.wav 2>/dev/null | head -n1 || true)"
  if [[ -n "$LATEST" && -f "$LATEST" ]]; then
    cp -f "$LATEST" "$OUT"
    exit 0
  else
    echo "ERROR: DiffRhythm ran but no WAV found in $OUT_DIR" >&2
    exit 2
  fi
fi

# Fallback: not found -> generate 1s silence so the pipeline still completes
echo "WARNING: diffrhythm not found; writing 1s of silence to $OUT for pipeline testing." >&2
python3 - "$OUT" <<'PY'
import wave, sys
with wave.open(sys.argv[1], 'w') as w:
    n_channels=1; sampwidth=2; framerate=16000; n_frames=16000
    w.setnchannels(n_channels); w.setsampwidth(sampwidth); w.setframerate(framerate)
    w.writeframes(b'\x00\x00'*n_frames)
PY

