#!/usr/bin/env bash
set -euo pipefail
REQ_JSON_PATH="$1"
MODEL="${MODEL:-qwen2.5:1.5b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama not found in PATH. Install it or export PATH accordingly." >&2
  exit 127
fi

PROMPT="$(python3 - "$REQ_JSON_PATH" <<'PY'
import json, sys
j=json.load(open(sys.argv[1], encoding='utf-8'))
theme=j.get('theme','')
emotion=j.get('emotion','')
style=j.get('style','现代流行')
lines=j.get('lines', 6)
constraints=j.get('constraints','尽量押韵；不要编号或引号；不要解释；每行一条。')
print(f"""请只输出中文歌词正文（每行一句，不要编号、不要引号、不要解释）。
主题：{theme}
情绪：{emotion}
风格：{style}
行数建议：{lines}
{constraints}""")
PY
)"

# Generate lyrics (trim empty lines)
ollama run "$MODEL" "$PROMPT" | awk 'NF'
