#!/usr/bin/env bash
set -euo pipefail
# Usage: gen_diffrhythm_runner.sh /abs/path/to/music_pipeline/data/requests/REQ-xxxxx
REQ_DIR="$1"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABS_REQ_DIR="$(cd "$REQ_DIR" && pwd)"

# Locate the DiffRhythm repo.
# Option A (recommended): set env DIFFRHYTHM_REPO="/abs/path/to/DiffRhythm"
# Option B: if the repo is checked out next to music_pipeline/, auto-detect it.
if [[ -n "${DIFFRHYTHM_REPO:-}" && -d "${DIFFRHYTHM_REPO}" ]]; then
  DIFF_DIR="${DIFFRHYTHM_REPO%/}"
elif [[ -d "$ROOT/../DiffRhythm" ]]; then
  DIFF_DIR="$(cd "$ROOT/../DiffRhythm" && pwd)"
else
  echo "ERROR: Cannot find DiffRhythm repo. Set DIFFRHYTHM_REPO=/abs/path/to/DiffRhythm" >&2
  exit 2
fi

# Read request.json for dynamic knobs
REF_PROMPT="$(python3 - <<PY
import json,sys,os
p=os.path.join("$ABS_REQ_DIR","request.json")
j=json.load(open(p,encoding="utf-8"))
print(j.get("ref_prompt","hopeful,ballad,pop,emotion"))
PY
)"

AUDIO_LEN="$(python3 - <<PY
import json,sys,os
p=os.path.join("$ABS_REQ_DIR","request.json")
j=json.load(open(p,encoding="utf-8"))
print(j.get("audio_length",95))
PY
)"

REPO_ID="${REPO_ID:-ASLP-lab/DiffRhythm-1_2}"     # can override via env on server
CHUNKED_FLAG="" ; [[ -n "${CHUNKED:-1}" ]] && CHUNKED_FLAG="--chunked"
BATCH_FLAG=""  ; [[ -n "${BATCH_INFER_NUM:-5}" ]] && BATCH_FLAG="--batch-infer-num ${BATCH_INFER_NUM:-5}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-0}"

OUT_DIR="$ABS_REQ_DIR/diffrhythm_output"
mkdir -p "$OUT_DIR"

RUNNER="$ABS_REQ_DIR/run_diffrhythm_req.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$DIFF_DIR"
export PYTHONPATH="\$PYTHONPATH:\$PWD"
export CUDA_VISIBLE_DEVICES=$CUDA_DEV

# macOS espeak note preserved from your original script
if [[ "\${OSTYPE:-}" =~ ^darwin ]]; then
  export PHONEMIZER_ESPEAK_LIBRARY=/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib
fi

PY="\${PYTHON:-python3}"
\$PY infer/infer.py \\
  --lrc-path "$ABS_REQ_DIR/final.lrc" \\
  --ref-prompt "$REF_PROMPT" \\
  --audio-length $AUDIO_LEN \\
  --repo-id "$REPO_ID" \\
  --output-dir "$OUT_DIR" \\
  $CHUNKED_FLAG \\
  $BATCH_FLAG

# Copy newest WAV to the canonical place the pipeline expects
LATEST=\$(ls -1t "$OUT_DIR"/*.wav 2>/dev/null | head -n1 || true)
if [[ -n "\$LATEST" ]]; then
  cp -f "\$LATEST" "$ABS_REQ_DIR/song.wav"
  echo "Wrote: $ABS_REQ_DIR/song.wav"
else
  echo "ERROR: No WAV found in $OUT_DIR" >&2
  exit 2
fi
EOF
chmod +x "$RUNNER"

echo "$RUNNER"
