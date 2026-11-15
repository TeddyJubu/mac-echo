# MacEcho

**A voice assistant for your Mac, and ears and a voice for your AI agents.**
Hearing and speaking run locally on Apple Silicon: your audio never leaves your machine.

MacEcho hears with FluidVoice's [Fluid Speech Pro](https://huggingface.co/FluidVoice-Labs/fluid-speech-pro-mlx),
the most accurate open speech-to-text model under 1B parameters, and speaks with Zyphra's
[ZONOS3-Mini](https://huggingface.co/mlx-community/ZONOS3-Mini), both through
[MLX](https://github.com/ml-explore/mlx). For your agents, the agent is the brain, so the agent CLI
needs no API key. The Mac app thinks with Claude Haiku, which needs an Anthropic API key and receives
the transcribed text (never the audio).

There are three ways to use it:

| | What you get | Start here |
|---|---|---|
| **Agent CLI** | `macecho listen` / `say` / `converse`: give Claude Code, Codex, Cursor or any script a voice conversation with you | [Quick start](#quick-start-voice-for-your-ai-agent) |
| **Mac app** | A Dock and menu bar app with a voice-reactive orb: press Return and talk | [Mac app](#mac-app) |
| **Python** | The streaming pipeline (VAD → Fluid Speech → Claude Haiku → ZONOS → speaker) to build on | [Development](#development) |

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
| `macecho show` | Opens the MacEcho window, which shows when the agent is listening and speaking. Loads no second model; needs the `[app]` extras. |
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
| `MACECHO_VOICE` | `imogen` | The ZONOS3-Mini voice: `imogen` or `cillian` read every test sentence word for word (`scripts/benchmark_speech.py`). `system` uses the macOS voice set in System Settings › Accessibility › Spoken Content. |
| `MACECHO_IDLE_MINUTES` | `30` | Minutes without a command before the daemon stops itself. `0` keeps it running. |
| `MACECHO_NO_SKILL` | unset | `1` stops the Claude Code skill from being installed automatically. |

Set these where the daemon starts: the environment of the first `macecho` command after the
daemon has stopped. Run `macecho stop` to apply a change.

### How it works

```
agent ──shell──▶ macecho ──Unix socket──▶ macecho daemon   (started on first use, stops when idle)
                                            ├─ Silero VAD   finds where your sentence ends
                                            ├─ Fluid Speech transcribes it;  ZONOS3-Mini reads replies aloud
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
rm -rf ~/.cache/huggingface/hub/models--FluidVoice-Labs--fluid-speech-pro-mlx ~/.cache/huggingface/hub/models--mlx-community--ZONOS3-Mini   # the models, ~1.5 GB
```

## Mac app

A Dock and menu bar app in which MacEcho itself is the assistant: you talk, Claude Haiku answers
in a ZONOS voice, and hands coding work in `MACECHO_PROJECT_DIR` to Claude Code. Set
`ANTHROPIC_API_KEY` before starting it.

- A dark window with an orb that moves with your voice and with each reply.
- Frequency bars under the orb.
- A Start/Stop button (Return presses it).
- A switch between the Imogen and Cillian voices.
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

Measured with `scripts/benchmark_speech.py` and `scripts/benchmark_brain.py` on a MacBook Air (M5).
Raw results are in `benchmarks/`.

| | |
|---|---|
| Transcribing what you said (Fluid Speech Pro, 8-bit) | median 140 ms |
| First sound of each spoken sentence (ZONOS3-Mini) | median 0.25 s |
| Claude Haiku's first sentence (the app) | median 1.35 s, mostly the network round trip |
| Models load and warm up at startup | about 6 s |
| Model memory | about 1 GB for both |

Timings vary with background load. Compare runs made back to back.

## Development

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[app]" pytest
PYTHONPATH=src .venv/bin/python -m pytest tests/
.venv/bin/python scripts/benchmark_speech.py # hearing and speaking; exits 1 if over budget
```

- **Sandboxes:** the CLI tests in `tests/test_agent_cli.py` bind Unix sockets, and the model needs
  the GPU (Metal). Both fail inside sandboxes that block those, such as some AI coding sandboxes;
  run them outside.
- **Layout:**
  - `src/macecho/cli.py`: the `macecho` command (standard library only);
  - `src/macecho/daemon/`: the daemon's server, engine and voices;
  - `src/macecho/speech_model.py`: transcription (Fluid Speech Pro) and speech (ZONOS3-Mini);
  - `src/macecho/brain.py` and `delegate.py`: the app's Claude Haiku brain and its Claude Code hand-overs;
  - `src/macecho/menubar.py` and `ui/`: the Mac app;
  - `src/macecho/agent.py`: the streaming pipeline the app uses.
- **Design docs:** `docs/superpowers/` holds the agent CLI's spec and implementation plan.

### The original pipeline

Originally, MacEcho chained SenseVoice speech recognition, a Qwen LLM via MLX and CosyVoice
text-to-speech. That code is still in `src/macecho/` (`asr/`, `llm/`, `tts/`) for reference, but the
app no longer uses it.

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
- Fluid Speech Pro by FluidVoice. Built on NVIDIA Parakeet TDT 0.6B v3 and Moondream Parakeet
  Ultra (CC BY 4.0), run with [parakeet-mlx](https://github.com/senstella/parakeet-mlx) and FluidVoice's
  loader (`src/macecho/fluid_speech_mlx.py`).
- [ZONOS3-Mini](https://huggingface.co/Zyphra/ZONOS3-Mini) by Zyphra (Apache 2.0), run with
  [mlx-audio](https://github.com/Neywa-Labs/mlx-audio) on Apple's [MLX](https://github.com/ml-explore/mlx).
- Claude Haiku and Claude Code by Anthropic; the brain's design borrows from
  [Bomby](https://github.com/bomby-app/bomby) (MIT).
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection.
- The window's orb and bar visualizer come from [ElevenLabs UI](https://github.com/elevenlabs/ui)
  (MIT).

## License

[MIT](LICENSE) for this fork's work: the speech integration, the Mac app, the agent CLI and the
rest of the changes made here. Code inherited from
[realtime-ai/mac-echo](https://github.com/realtime-ai/mac-echo) remains under its authors' terms.
That project's README states MIT, but it ships no license file.
