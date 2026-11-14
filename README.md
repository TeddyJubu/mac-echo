# MacEcho

**A private voice assistant for your Mac, and ears and a voice for your AI agents.**
Everything runs locally on Apple Silicon: no cloud, no API keys, no audio leaves your machine.

MacEcho is built on Liquid AI's [LFM3](https://huggingface.co/LiquidAI) end-to-end speech models
running through [MLX](https://github.com/ml-explore/mlx). One model hears you and talks back, so
there is no separate speech recognition, LLM and text-to-speech chain to wire together.

There are three ways to use it:

| | What you get | Start here |
|---|---|---|
| **Agent CLI** | `macecho listen` / `say` / `converse`: give Claude Code, Codex, Cursor or any script a voice conversation with you | [Quick start](#quick-start-voice-for-your-ai-agent) |
| **Mac app** | A Dock and menu bar app with a voice-reactive orb: press Return and talk | [Mac app](#mac-app) |
| **Python** | The streaming pipeline (VAD → LFM3 → speaker) to build on | [Development](#development) |

## Requirements

- A Mac with Apple Silicon (M-series). MLX does not run on Intel Macs.
- Python 3.12 or 3.13. Newer versions are not supported yet: onnxruntime, used for voice
  detection, has no wheels for them.
- About 2 GB of disk space and 2 GB of memory for the speech model.
- [uv](https://docs.astral.sh/uv/) to install the CLI (`brew install uv`).

## Quick start: voice for your AI agent

```bash
uv tool install git+https://github.com/TeddyJubu/mac-echo
macecho say "Hello, I can talk now."
```

The first command downloads the speech model (about 1–2 GB, once), loads it into a small
background daemon, and speaks. After that, commands respond immediately.

```bash
macecho listen                                   # wait for one spoken sentence, print it
macecho say "The tests pass."                    # speak, return when finished
reply=$(macecho converse "Done. What next?")     # speak, then listen: one call per turn
macecho status                                   # model, voice and audio devices, as JSON
macecho stop                                     # stop the daemon, free ~2 GB of memory
```

The first time something listens, macOS asks whether **your terminal app** (Terminal, iTerm,
VS Code…) may use the microphone. Allow it.

### With Claude Code

If Claude Code is installed, the first `macecho` command also installs a `macecho-voice` skill in
`~/.claude/skills/`. In any Claude Code session, say **"let's talk"** or **"voice mode"** and it
starts a spoken conversation: it listens to you, does the work, and answers out loud.

- To skip the automatic install, set `MACECHO_NO_SKILL=1`.
- To refresh the skill after an upgrade, run `macecho install-skill --force`.

### With any other agent

Point the agent at `macecho guide`. It prints the conversation loop, the exit codes and how to
phrase spoken replies, written for an agent to follow. This repository's `AGENTS.md` does the same
for agents that read it automatically. The whole loop is one command per turn:

```bash
reply=$(macecho converse "I'm listening. What should we work on?")
# …act on "$reply" as if the user had typed it, then answer and listen again:
reply=$(macecho converse "Done. The build passes. Anything else?")
```

### Command reference

| Command | What it does |
|---|---|
| `macecho listen [--timeout S] [--json]` | Opens the mic, waits for one complete sentence, prints exactly what you said. |
| `macecho say "text"` | Speaks the text, which can also be piped on stdin. Markdown, code blocks and links are dropped first. |
| `macecho converse "text" [--timeout S] [--json]` | `say`, then `listen`, in one call. |
| `macecho status` | JSON: running, pid, model, voice, input and output device, idle minutes left. Never starts the daemon. |
| `macecho stop` | Stops the daemon and frees the model's memory. |
| `macecho guide` | Prints the guide for agents. |
| `macecho install-skill [--force]` | Installs, or with `--force` refreshes, the Claude Code skill. |

stdout carries only the result, so agents can capture it directly. Progress and errors go to
stderr. `--json` prints `{"text": "...", "seconds": 2.4}`.

| Exit code | Meaning |
|---|---|
| `0` | Success. |
| `1` | Any other error. The message is on stderr; if the daemon crashed, it includes the log path. |
| `2` | Nobody spoke before `--timeout` (default 120 s). stdout is empty. |
| `3` | Microphone access is off for the app running the command. Turn it on in System Settings › Privacy & Security › Microphone. |

### Configuration

All settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MACECHO_SPEECH_MODEL` | `LiquidAI/LFM3-Audio-2B-MLX-4bit` | The speech model. `LiquidAI/LFM3-Omni-2B-MLX-8bit` also works. |
| `MACECHO_VOICE` | `lfm3` | `lfm3` speaks in LFM3's own voice. `system` uses the macOS voice set in System Settings › Accessibility › Spoken Content. |
| `MACECHO_IDLE_MINUTES` | `30` | Minutes without a command before the daemon stops itself. `0` keeps it running. |
| `MACECHO_NO_SKILL` | unset | `1` stops the Claude Code skill from being installed automatically. |

Set these where the daemon starts: the environment of the first `macecho` command after the
daemon has stopped. Run `macecho stop` to apply a change.

### How it works

```
agent ──shell──▶ macecho ──Unix socket──▶ macecho daemon   (started on first use, stops when idle)
                                            ├─ Silero VAD   finds where your sentence ends
                                            ├─ LFM3         transcribes it, and reads replies aloud
                                            ├─ microphone   open only while listening
                                            └─ speaker      re-checked each time (plug headphones in any time)
```

The `macecho` command is small and starts instantly. The daemon keeps the model in memory between
turns and handles one request at a time, so two agents can share it. It lives in
`~/Library/Application Support/MacEcho/`, and its log is `~/Library/Logs/MacEcho-daemon.log`.

### Uninstall

```bash
macecho stop
uv tool uninstall macecho
rm -rf ~/.claude/skills/macecho-voice "$HOME/Library/Application Support/MacEcho" ~/Library/Logs/MacEcho-daemon.log
rm -rf ~/.cache/huggingface/hub/models--LiquidAI--LFM3-Audio-2B-MLX-4bit   # the model, ~2 GB
```

## Mac app

A Dock and menu bar app in which MacEcho itself is the assistant: you talk, and LFM3 answers.

- A dark window with an orb that moves with your voice and with each reply.
- Frequency bars under the orb.
- A Start/Stop button (Return presses it).
- A switch between the Audio-2B and Omni-2B models.
- The same controls in the menu bar.

The app runs from a checkout of this repository:

```bash
git clone https://github.com/TeddyJubu/mac-echo && cd mac-echo
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[app]"
scripts/build_app.sh        # builds the window's page (needs Node) and dist/MacEcho.app
open dist/MacEcho.app
```

- **Keep it in the Dock:** right-click the Dock icon, then Options › Keep in Dock.
- **Closing the window** leaves MacEcho running; click the Dock icon to bring it back.
- **After changing code,** quit with ⌘Q and reopen. `scripts/build_app.sh` quits a running copy
  for you.
- **Menu bar icons:** 🎙… loading · 🎙 off · 🎙● listening · 🎙🔊 replying · 🎙⚠️ error (see
  Open Log, which opens `~/Library/Logs/MacEcho.log`).
- **If the orb reacts to room noise or barely moves when you speak,** adjust `LEVEL_FLOOR_DB` and
  `LEVEL_CEIL_DB` in `src/macecho/device/device.py`.

## Performance

Measured with `scripts/benchmark.py` on a MacBook Air (Apple Silicon) running LFM3-Audio-2B,
4-bit. Raw results are in `benchmarks/`.

| | |
|---|---|
| First reply audio after you stop speaking | 0.24–0.32 s, plus the 0.6 s pause that marks the end of your sentence |
| Model load at startup | 2.1–2.5 s |
| Spoken replies | generated 2.4–3× faster than they play |
| Model memory | about 1.8 GB |

Timings vary with background load. Compare runs made back to back.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[app]" pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/
.venv/bin/python scripts/benchmark.py        # latency benchmark; exits 1 if over budget
```

- **Sandboxes:** the CLI tests in `tests/test_agent_cli.py` bind Unix sockets, and the model needs
  the GPU (Metal). Both fail inside sandboxes that block those, such as some AI coding sandboxes;
  run them outside.
- **Layout:**
  - `src/macecho/cli.py`: the `macecho` command (standard library only);
  - `src/macecho/daemon/`: the daemon's server, engine and voices;
  - `src/macecho/speech_model.py`: LFM3 chat, transcription and text-to-speech;
  - `src/macecho/menubar.py` and `ui/`: the Mac app;
  - `src/macecho/agent.py`: the streaming pipeline the app uses.
- **Design docs:** `docs/superpowers/` holds the agent CLI's spec and implementation plan.

### The original pipeline

Before LFM3, MacEcho chained SenseVoice speech recognition, a Qwen LLM via MLX and CosyVoice
text-to-speech. That code is still in `src/macecho/` (`asr/`, `llm/`, `tts/`). It is used when
`MACECHO_SPEECH_MODEL` is not set, and needs the heavier dependencies in `requirements.txt`. The
LFM3 path described above is the maintained one.

## Troubleshooting

- **No sound from replies.** Replies go to the current default output device, checked each time
  you start listening. If you switched outputs mid-session, stop and start again. If that doesn't
  help, run `macecho status` to see which output the daemon is using.
- **Exit code 3 / "Microphone access is off".** Allow your terminal app in System Settings ›
  Privacy & Security › Microphone, then quit and reopen the terminal.
- **`listen` returns something nobody said.** A short noise can occasionally be transcribed as a
  word. Every transcription is logged with its length and loudness in
  `~/Library/Logs/MacEcho-daemon.log`. Please include those lines in a bug report.
- **The first command takes a long time.** The model is downloading (once) or loading. The
  download goes to `~/.cache/huggingface/`; set `HF_HOME` to put it elsewhere.
- **Anything else.** Run `macecho stop` and try again, and check
  `~/Library/Logs/MacEcho-daemon.log`.

## Credits

- [realtime-ai/mac-echo](https://github.com/realtime-ai/mac-echo), the original project this fork
  builds on.
- [LFM3](https://huggingface.co/LiquidAI) speech models by Liquid AI, run with
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) on Apple's [MLX](https://github.com/ml-explore/mlx).
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection.
- The window's orb and bar visualizer come from [ElevenLabs UI](https://github.com/elevenlabs/ui)
  (MIT).

## License

[MIT](LICENSE) for this fork's work: the LFM3 integration, the Mac app, the agent CLI and the
rest of the changes made here. Code inherited from
[realtime-ai/mac-echo](https://github.com/realtime-ai/mac-echo) remains under its authors' terms.
That project's README states MIT, but it ships no license file.
