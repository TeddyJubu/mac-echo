# MacEcho agent CLI: ears and voice for AI agents

Date: 2026-11-14 · Status: design approved in conversation, awaiting spec review

## Goal

Give any AI agent that can run shell commands (Claude Code, Codex, Cursor, scripts) a voice
interface on a Mac. The agent stays the brain. MacEcho listens to the user and returns their
words as text, and speaks the agent's text aloud. After one install command it works with no
further setup.

**Success:** an agent runs one command, the user speaks, and the agent gets the sentence back
within about a second of the user finishing. The agent runs another command and the user hears
the reply.

### Out of scope for v1

- Barge-in (interrupting playback by speaking). Without echo cancellation, speaker output would
  trigger the mic. Revisit together with echo cancellation.
- An MCP server front-end. It can sit on the same daemon later.
- Sharing one model between the menu bar app and the daemon. In v1 they are independent; if both
  run, each loads its own copy.
- Platforms other than Apple Silicon macOS.

## Command surface

```
macecho listen   [--timeout SECS] [--json]
macecho say      "text"                      # or text on stdin
macecho converse "text" [--timeout SECS] [--json]
macecho status
macecho stop
macecho guide
macecho install-skill [--force]
```

| Command | Behaviour |
|---|---|
| `listen` | Opens the mic and waits until the user finishes one utterance (Silero VAD, 0.6 s end-of-speech). Transcribes it and prints the text to stdout. `--timeout` defaults to 120 s. `--json` prints `{"text": str, "seconds": float}` instead of bare text. |
| `say` | Speaks the text. Returns after playback finishes, so a following `listen` never hears the agent's own voice. |
| `converse` | `say`, then `listen`, in one call. Prints the transcript. One agent turn becomes one tool call. |
| `status` | JSON: daemon running, pid, model repo, whether the model is loaded, input and output device names, idle minutes left. Does not start the daemon. |
| `stop` | Shuts the daemon down and frees the model memory (about 1.9 GB). |
| `guide` | Prints the agent-facing usage guide (see Agent guide below). |
| `install-skill` | Installs the Claude Code skill. `--force` overwrites an existing copy. |

### Output contract

- stdout carries only the result: the transcript, or JSON where documented. Never progress text.
- stderr carries progress ("Loading voice model…") and errors.
- Exit codes:
  - `0`: success.
  - `1`: other error, with a message and the daemon log path on stderr.
  - `2`: no speech before `--timeout`. stdout is empty.
  - `3`: microphone permission denied. stderr names the app to allow in System Settings ›
    Privacy & Security › Microphone.

## Architecture

```
agent ──shell──▶ macecho (client) ──Unix socket, JSON──▶ macecho daemon
                                                          ├─ SpeechModel (LFM3, loaded once)
                                                          ├─ Silero VAD
                                                          ├─ mic   (opened per listen)
                                                          └─ voice (LFM3 audio mode or macOS say)
```

### Units

Each unit has one job and is testable on its own.

- **`macecho/cli.py`: client and argument parsing.** Ensures a daemon is running, sends one
  request, maps the reply to stdout, stderr and an exit code. Imports no MLX, so it starts fast
  and its tests need no GPU.
- **`macecho/daemon/server.py`: socket server.** Accepts connections on
  `~/Library/Application Support/MacEcho/daemon.sock`. Reads one JSON request line and writes one
  JSON reply line. Requests are serialised with a lock: a second caller waits its turn. Also owns
  the idle timer.
- **`macecho/daemon/engine.py`: voice engine.** `listen(timeout) -> Utterance | None`,
  `say(text) -> None`, `status() -> dict`. Wraps `SpeechModel`, `VadProcessor`, `AudioRecorder`
  and the voice. This is the only unit that touches audio hardware or MLX. The server depends
  only on its interface, so tests substitute a fake.
- **`macecho/daemon/voice.py`: text to speech.** `LfmVoice` (LFM3 audio-only generation mode)
  and `SystemVoice` (macOS `say`), behind one `speak(text)` method. Selected by
  `MACECHO_VOICE=lfm3|system`.
- **`macecho/agent_guide.md`: the agent guide.** Shipped as package data. `guide` prints it,
  and the skill and `AGENTS.md` reuse it.

### Protocol

Request: `{"op": "listen", "timeout": 120}`, `{"op": "say", "text": "..."}`,
`{"op": "converse", "text": "...", "timeout": 120}`, `{"op": "status"}` or `{"op": "stop"}`.

Reply: `{"ok": true, "text": "...", "seconds": 2.4}`, or
`{"ok": false, "error": "timeout" | "mic_denied" | "internal", "message": "..."}`.

The client maps `timeout` to exit 2, `mic_denied` to exit 3 and anything else to exit 1.

### Daemon lifecycle

- **Auto-start.** If the socket is missing, or connecting is refused, the client removes any
  stale socket file and spawns `python -m macecho.daemon` detached (`start_new_session=True`). Its
  stdout and stderr go to `~/Library/Logs/MacEcho-daemon.log`. The client polls the socket for up
  to 10 minutes; the first run may download the model. Until the daemon is up, the client prints
  `Loading voice model…`, or `Downloading voice model (first run only, ~1–2 GB)…` when the model
  is not yet in the Hugging Face cache.
- **Single instance.** The daemon takes an exclusive `fcntl.flock` on `daemon.lock` before
  binding. A second daemon started in a race exits quietly, and its client connects to the
  winner.
- **Idle stop.** After `MACECHO_IDLE_MINUTES` minutes (default 30) without a request, the daemon
  closes the socket, removes it and exits. The next command restarts it, at about 3 s cost.
- **Crash recovery.** If a request fails at the socket level (connection reset, or EOF without a
  reply), the client restarts the daemon once and retries. If the retry also fails, it exits `1`
  and prints the log path.

### Listening

1. Call `device.refresh_devices()`, so headphones plugged in since the last request are used.
   Then open an `AudioRecorder` at 16 kHz. The macOS mic indicator is lit only while a
   `listen` is in progress.
2. Feed 32 ms frames to a fresh `VadProcessor` until `SPEECH_END` or the timeout.
3. Close the recorder, then `SpeechModel.transcribe(segment)`: ASR mode, language detected
   automatically.
4. Empty or whitespace-only text (a cough or click that the VAD took for speech) is discarded and
   listening resumes within the same timeout.
5. **Permission check.** When the app has no microphone access, macOS does not raise an error; it
   delivers digital silence. So if the first 1.0 s of mic frames are all exactly zero, the result
   is `mic_denied`. A real room never produces exact zeros. A `PortAudioError` when opening the
   stream is reported as `internal`, with its message.

### Speaking

1. `refresh_devices()`, then open an `AudioPlayer` at the voice's sample rate.
2. **`LfmVoice`**: run LFM3 in audio-only generation mode with a TTS prompt. Stream the 24 kHz
   chunks to the player as they are decoded.
3. **`SystemVoice`**: `subprocess.run(["say", text])`. It blocks until speech ends and uses the
   voice chosen under System Settings › Spoken Content.
4. Return only after playback has drained.

**Voice selection.** The first implementation step is a feasibility test of LFM3 audio-only
mode on 5 fixed sentences. It passes if:

- transcribing the generated audio back with `SpeechModel.transcribe` gives at most 1 word
  error per sentence; and
- the median time to first audio is at most 1.0 s.

If it passes, the default is `lfm3`; otherwise the default is `system`. Either way,
`MACECHO_VOICE` overrides the default. The result is recorded in this spec.

### Configuration

All settings are environment variables, read through the existing `MacEchoConfig` where it
applies.

| Variable | Default | Meaning |
|---|---|---|
| `MACECHO_SPEECH_MODEL` | `LiquidAI/LFM3-Audio-2B-MLX-4bit` | Model repo. |
| `MACECHO_VOICE` | set by the feasibility test | `lfm3` or `system`. |
| `MACECHO_IDLE_MINUTES` | `30` | Idle stop. `0` disables it. |
| `MACECHO_NO_SKILL` | unset | `1` stops the skill from being installed automatically. |

## Packaging and install

- Replace `setup.py` with `pyproject.toml`: name `macecho`, Python `>=3.12,<3.14` (onnxruntime
  has no wheels for newer versions), and `[project.scripts] macecho = "macecho.cli:main"`.
- Core dependencies are the voice path only:
  - `mlx-vlm==0.7.12` (the tested version),
  - `sounddevice`, `numpy`, `onnxruntime`, `pydantic`, `pydantic-settings`.
- Optional extra `[app]` holds the menu bar app's `rumps` and `pyobjc-framework-WebKit`.
  `requirements.txt` keeps the legacy torch/funasr stack and is not used by this install.
- Package data: `vad/silero_vad.onnx`, `agent_guide.md` and `skill/SKILL.md`.
- Install:

  ```
  uv tool install git+https://github.com/TeddyJubu/mac-echo
  ```

  `pipx install git+…` works the same way.

## Agent guide and skill

`agent_guide.md`, kept under 60 lines, covers:

- the conversation loop:

  ```
  reply=$(macecho converse "<short spoken answer>")
  ```

  On exit 2, the agent says nothing and calls `macecho listen` again, or stops if the user has
  gone quiet.
- spoken-style rules: short sentences, no markdown, code or long lists. Show those on screen and
  say "I've put it on screen";
- the exit codes, and that `macecho stop` frees memory when the conversation ends.

**Claude Code skill: `macecho-voice`.**

- Its description triggers on "voice mode", "let's talk" and "talk to me". Its body is the
  agent guide.
- **Installed automatically** on the first run of any `macecho` command, if all of these hold:
  - `~/.claude/` exists;
  - `~/.claude/skills/macecho-voice/SKILL.md` does not exist;
  - `MACECHO_NO_SKILL` is not set.
- When it installs the skill, MacEcho prints one line to stderr:
  `Installed Claude Code skill: ~/.claude/skills/macecho-voice (set MACECHO_NO_SKILL=1 to skip)`.
- It never overwrites an existing copy; `macecho install-skill --force` does.
- To remove it, delete the folder.

`AGENTS.md` at the repo root gains a short "Voice" section that points to `macecho guide`, for
Codex and other agents that read it.

## Testing

**Automated tests** need no GPU and no mic, and run in the sandbox:

- **Protocol and server:** run the real socket server with a fake engine. Check round trips for
  every op, that requests are serialised, and the idle stop with an injected clock.
- **Client lifecycle:**
  - A stale socket file is removed and the daemon respawned (spawn is injected as a fake).
  - A crash is retried exactly once.
  - The lock prevents a second daemon.
- **Output contract:** for each error kind, check the exit code, and that stdout holds only the
  result.
- **Listening logic:** the real `VadProcessor` fed from a WAV fixture (silence, speech, silence)
  yields one segment. A cough-only fixture yields nothing and keeps waiting. The transcriber is
  faked.
- **Skill auto-install:** installs when absent; does nothing when present; honours
  `MACECHO_NO_SKILL`; `--force` overwrites. Uses a temporary `HOME`.

**Manual checks** need the sandbox off, the GPU and audio:

- **End to end:** generate a sentence with `say -o`, feed it through the daemon's engine and
  assert that the transcript matches; time a second `listen` with the model warm.
- **Voice feasibility test:** see Speaking.
- **Install:** `uv tool install` from the fork into a clean environment, then run
  `macecho status`, `say` and `listen`.

## Risks

- **LFM3 TTS fidelity is unknown.** Mitigated by the feasibility test and the system-voice
  fallback.
- **Microphone permission belongs to the agent's terminal app,** not MacEcho, so the first
  `listen` triggers that app's prompt. Exit 3 explains what to do.
- **Two copies of the model** if the menu bar app and the daemon run together (about 3.8 GB).
  Documented. Sharing is a later step.
