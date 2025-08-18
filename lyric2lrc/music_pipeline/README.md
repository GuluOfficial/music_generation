# music_pipeline (local-first)
End-to-end local pipeline for: **lyrics (Chinese)** → **heuristic timestamps** → **strict LRC** → **DiffRhythm infer**.

## Layout
```
music_pipeline/
  bin/            # shell entrypoints
  py/             # Python utilities (stdlib only)
  env/            # venv setup
  data/requests/  # per-request artifacts (lyrics, LRCs, audio)
  data/latest -> requests/REQ-xxxxx
```

## Quick start
1) (Optional) Create venv:
```bash
bash env/setup.sh
```
2) Ensure **Ollama** is installed and a small Chinese model is available:
```bash
ollama pull qwen2.5:1.5b-instruct
```
3) (Optional) Ensure DiffRhythm is reachable. If it's a custom path, set:
```bash
export DIFFRHYTHM_CMD="/absolute/path/to/diffrhythm_runner"
```
4) Run a demo request:
```bash
bash bin/pipeline.sh "$(cat demo_request.json)"
```
The output folder is printed at the end and also symlinked at `data/latest/`.

## Request JSON schema (minimal)
```json
{
  "theme": "夏夜与河流",
  "emotion": "温柔内省",
  "style": "现代流行",
  "lines": 6,
  "constraints": "尽量押韵；不要编号；不要引号；不要解释；每行一条。"
}
```

## Strict LRC rules (so DiffRhythm accepts it)
- One timestamp per line.
- Format: `[mm:ss.cc]` with dot as decimal and **two** digits.
- **UTF-8 (no BOM)**, newline = `\n`.
- Times strictly increasing (nudged by 0.01s if needed).

## Tunable pacing
`py/lyrics2lrc.py` supports flags like:
```
--base 0.8 --per-char 0.22 --min 1.8 --max 6.0 --gap 0.35
```

## Fallback audio
If DiffRhythm isn’t found, the pipeline writes a 1s silent WAV so you can still validate the chain.

## Notes
- You can pipe other lyric sources—`bin/gen_lyrics_ollama.sh` just reads `request.json` and prints plain lines.
- To integrate with your server, call `bin/pipeline.sh` from your Flask handler and return the artifact paths.
