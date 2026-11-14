# CLAUDE.md

Guidance for Claude Code when working in this repository. User-facing docs are in `README.md`;
`AGENTS.md` tells agents how to *use* MacEcho's voice. This file is for *changing* it.

## What MacEcho is

A local voice assistant for Apple Silicon, built on Liquid AI's LFM3 end-to-end speech models
(`LiquidAI/LFM3-Audio-2B-MLX-4bit` by default, or `LFM3-Omni-2B-MLX-8bit`) via `mlx-vlm`. One
model hears, thinks and speaks; there is no separate ASR → LLM → TTS chain on the maintained path.

Three ways in, sharing the same core:

| Entry point | Code | Notes |
|---|---|---|
| **Agent CLI** `macecho listen/say/converse/status/stop/guide/install-skill` | `src/macecho/cli.py` | Standard library only, so it starts instantly. Auto-starts the daemon. |
| **Daemon** | `src/macecho/daemon/` (`server.py`, `protocol.py`, `engine.py`, `voice.py`, `__main__.py`) | Holds the model. Unix socket in `~/Library/Application Support/MacEcho/` (`MACECHO_RUNTIME_DIR` overrides). One request at a time. Stops after `MACECHO_IDLE_MINUTES` (30). Log: `~/Library/Logs/MacEcho-daemon.log`. |
| **Mac app** (Dock window + menu bar) | `src/macecho/menubar.py`, `ui/` (Vite/React page, built into `src/macecho/ui/index.html`) | Drives `agent.py`. Frees the model after 30 idle minutes (`should_unload`). Log: `~/Library/Logs/MacEcho.log`. |

Core modules:
- `speech_model.py`: `SpeechModel`, LFM3 chat, transcription (`Perform ASR.`) and TTS (`Perform TTS.`).
- `agent.py`: the streaming pipeline the app uses (mic → 32 ms frames → VAD → LFM3 → speaker).
- `vad/`: Silero VAD (`silero_vad.onnx`, onnxruntime, single-threaded on purpose).
- `device/device.py`: `AudioRecorder`/`AudioPlayer` (sounddevice), plus `level`/`bands` for the orb.
  Audio devices are re-read at the start of every session.
- `config.py`: Pydantic Settings (`MACECHO_` env vars, `.env`, defaults).
- `agent_guide.md`: what `macecho guide` prints and the Claude Code skill installs.

**Legacy:** `asr/` (SenseVoice), `llm/` (Qwen via MLX, OpenAI), `tts/` (CosyVoice, Kokoro),
`sentencizer/`, `message/`. They are only used when `MACECHO_SPEECH_MODEL` is unset and need the
heavy deps in `requirements.txt`. Don't extend them; the LFM3 path is the maintained one.

## Commands

```bash
uv venv --python 3.12 .venv                              # 3.12/3.13 only: onnxruntime has no newer wheels
uv pip install --python .venv/bin/python -e ".[app]" pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/         # tests
.venv/bin/python scripts/benchmark.py                    # latency benchmark, exits 1 over budget
scripts/build_app.sh                                     # builds ui/ (needs Node) and dist/MacEcho.app
PYTHONPATH=src .venv/bin/python -m macecho.menubar       # run the app without bundling
uv tool install --force .                                # install the CLI from this checkout
```

Design docs for the CLI live in `docs/superpowers/` (spec and plan).

## Gotchas

- **Sandboxes:** the CLI tests bind Unix sockets, MLX needs Metal, and Hugging Face downloads need
  the network. All three fail inside the Claude Code sandbox; run those with it off.
- **LFM3 prompts are triggers, not suggestions.** It only chats if the system prompt starts with
  `Respond with interleaved text and audio.` and `generation_mode="interleaved"` is set; otherwise
  it transcribes or echoes. `Perform ASR.` / `Perform TTS.` select transcription and speech.
- **Freeing the model** needs `gc.collect()` and `mx.clear_cache()`; macOS then reclaims the GPU
  memory asynchronously (~3 s). Measure the footprint after a pause, or it looks like a leak.
- **Benchmarks** swing with background load. Compare runs made back to back, not against old JSON.
- **No echo cancellation yet:** on speakers the mic hears the replies; use headphones to test.
