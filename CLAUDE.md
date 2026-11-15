# CLAUDE.md

Guidance for Claude Code when working in this repository. User-facing docs are in `README.md`;
`AGENTS.md` tells agents how to *use* MacEcho's voice. This file is for *changing* it.

## What MacEcho is

A voice assistant for Apple Silicon. Fluid Speech Pro hears (`FluidVoice-Labs/fluid-speech-pro-mlx`,
via parakeet-mlx and the vendored `fluid_speech_mlx.py`) and ZONOS3-Mini speaks
(`mlx-community/ZONOS3-Mini`, via mlx-audio pinned to a git commit), both on MLX. In the agent CLI the
calling agent thinks; in the Mac app Claude Haiku does (`brain.py`).

Three ways in, sharing the same core:

| Entry point | Code | Notes |
|---|---|---|
| **Agent CLI** `macecho listen/say/converse/status/stop/guide/install-skill` | `src/macecho/cli.py` | Standard library only, so it starts instantly. Auto-starts the daemon. |
| **Daemon** | `src/macecho/daemon/` (`server.py`, `protocol.py`, `engine.py`, `voice.py`, `__main__.py`) | Holds the model. Unix socket in `~/Library/Application Support/MacEcho/` (`MACECHO_RUNTIME_DIR` overrides). One request at a time. Stops after `MACECHO_IDLE_MINUTES` (30). Log: `~/Library/Logs/MacEcho-daemon.log`. |
| **Mac app** (Dock window + menu bar) | `src/macecho/menubar.py`, `ui/` (Vite/React page, built into `src/macecho/ui/index.html`) | Drives `agent.py`. Frees the model after 30 idle minutes (`should_unload`). Log: `~/Library/Logs/MacEcho.log`. |

Core modules:
- `speech_model.py`: `SpeechModel`: `transcribe()` (16 kHz in) and `tts()` (24 kHz chunks out, resampled
  from ZONOS's 44.1 kHz). One lock serialises the two models.
- `agent.py`: the streaming pipeline the app uses (mic → 32 ms frames → VAD → Fluid Speech → Haiku → ZONOS → speaker).
- `brain.py`: Claude Haiku thinks for the app and hands work to Claude Code (`delegate.py`) in
  `MACECHO_PROJECT_DIR` through `start_task`, and stops it through `stop_task`. Needs
  `ANTHROPIC_API_KEY`. `scripts/benchmark_brain.py` replays routing cases and past failed dialogues.
- `vad/`: Silero VAD (`silero_vad.onnx`, onnxruntime, single-threaded on purpose).
- `device/device.py`: `AudioRecorder`/`AudioPlayer` (sounddevice), plus `level`/`bands` for the orb.
  Audio devices are re-read at the start of every session.
- `config.py`: Pydantic Settings (`MACECHO_` env vars, `.env`, defaults).
- `agent_guide.md`: what `macecho guide` prints and the Claude Code skill installs.

## Commands

```bash
uv venv --python 3.12 .venv                              # 3.12/3.13 only: onnxruntime has no newer wheels
uv pip install --python .venv/bin/python -e ".[app]" pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/         # tests
.venv/bin/python scripts/benchmark_speech.py             # hearing + speaking round trip, exits 1 over budget
scripts/build_app.sh                                     # builds ui/ (needs Node) and dist/MacEcho.app
PYTHONPATH=src .venv/bin/python -m macecho.menubar       # run the app without bundling
uv tool install --force .                                # install the CLI from this checkout
```

Design docs for the CLI live in `docs/superpowers/` (spec and plan).

## Gotchas

- **Sandboxes:** the CLI tests bind Unix sockets, MLX needs Metal, and Hugging Face downloads need
  the network. All three fail inside the Claude Code sandbox; run those with it off.
- **Not every ZONOS voice is reliable.** Some repeat the sentence or babble on after it (wyatt, rupert,
  savannah). Check a new voice with `scripts/benchmark_speech.py --voice NAME` before offering it.
- **mlx-audio is pinned to a git commit**: ZONOS3 support isn't in a PyPI release yet. Move to a
  release once one includes `mlx_audio/tts/models/zonos3`.
- **Freeing the model** needs `gc.collect()` and `mx.clear_cache()`; macOS then reclaims the GPU
  memory asynchronously (~3 s). Measure the footprint after a pause, or it looks like a leak.
- **Benchmarks** swing with background load. Compare runs made back to back, not against old JSON.
- **Echo cancellation** (`device/voice_io.py`, Apple's voice processing) runs only on the Mac's own
  speakers. With headphones it's skipped: it would move the input to the headphone jack.
