# MacEcho agent CLI: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `macecho` command that gives any shell-capable AI agent ears (`listen`) and a voice
(`say`), backed by an auto-started daemon that keeps LFM3 loaded.

**Architecture:**
- **Client (`cli.py`):** standard library only. It talks one JSON line each way over a Unix
  socket, starting the daemon on demand.
- **Daemon:** a small socket server (`server.py`) in front of an `Engine` (`engine.py`) that owns
  the model, VAD, mic and voice.
- **Tests:** the server and client are tested against a fake engine; the listening logic is
  tested with the real VAD on a WAV file.

**Tech stack:** Python 3.12, stdlib `socket`, `fcntl`, `argparse` and `subprocess`; existing
`SpeechModel`, `VadProcessor`, `AudioRecorder`, `AudioPlayer` and `refresh_devices`; mlx-vlm
0.7.12.

**Spec:** `docs/superpowers/specs/2026-11-14-macecho-agent-cli-design.md`

## Global constraints

- Python `>=3.12,<3.14`; `mlx-vlm==0.7.12`.
- stdout holds only results. Progress and errors go to stderr.
- Exit codes: 0 OK, 1 error, 2 no speech before the timeout, 3 microphone denied.
- Socket `~/Library/Application Support/MacEcho/daemon.sock`, lock `daemon.lock` next to it, log
  `~/Library/Logs/MacEcho-daemon.log`. `MACECHO_RUNTIME_DIR` overrides the socket and lock
  directory, for tests.
- Environment: `MACECHO_SPEECH_MODEL`; `MACECHO_VOICE` (default `lfm3`);
  `MACECHO_IDLE_MINUTES` (default 30, `0` turns the idle stop off); `MACECHO_NO_SKILL`.
- `cli.py` and `daemon/protocol.py` must not import MLX, numpy or sounddevice.

## Review focus

Inputs the spec implies but the obvious tests miss, each pinned by a test in its owning task:

1. **Spoken text full of markdown or code**, for example "`npm test` **passed**". `say` should
   speak plain words, not the symbols. *Test: `SystemVoice`/`LfmVoice` speak `speakable(text)`;
   Task 4.*
2. **Empty `say` text** (an empty argument or empty stdin). Exit 1 with "nothing to say", without
   starting the daemon. *Test: Task 2.*
3. **A request larger than one socket read**, such as a 20 KB `say` text. The full text must
   arrive. *Test: Task 1, round trip of 20 000 characters.*
4. **The client is killed (Ctrl-C) during `listen`.** The daemon must stay usable for the next
   request. *Test: Task 1; a client that disconnects mid-request does not wedge the lock.*
5. **Non-UTF-8 or malformed JSON request.** The server replies `internal` and keeps serving.
   *Test: Task 1.*

---

### Task 1: Protocol and socket server

**Files:**
- Create: `src/macecho/daemon/__init__.py` (empty)
- Create: `src/macecho/daemon/protocol.py`
- Create: `src/macecho/daemon/server.py`
- Test: `tests/test_agent_cli.py`

**Interfaces it produces:**
- `protocol.runtime_dir() -> Path`; `protocol.SOCKET`, `LOCK`, `LOG`: functions returning paths.
- `protocol.send(sock, obj: dict) -> None`; `protocol.recv(sock) -> dict | None`. One
  newline-terminated JSON line each way.
- `protocol.ERRORS = {"timeout": 2, "mic_denied": 3}`. Anything else maps to exit 1.
- `server.serve(engine, idle_minutes: float, clock=time.monotonic) -> None`. Binds the socket
  under the lock and serves one request at a time. Exits on `stop` or after the idle timeout.
- `server.handle(engine, request: dict) -> dict`. Engine contract: `listen(timeout) -> dict`,
  `say(text) -> None`, `status() -> dict`. Engine methods raise `EngineError(kind, message)`.

**Steps:**
- [ ] **Tests.** Using `FakeEngine` and a real server thread on a short `MACECHO_RUNTIME_DIR`:
  - a round trip for each op: listen, say, converse, status, stop;
  - converse calls say, then listen;
  - `EngineError("timeout")` produces a reply with `ok: false, error: "timeout"`;
  - a 20 000-character say text arrives whole;
  - malformed JSON produces `internal` and the server still answers the next request;
  - a client that disconnects mid-request does not block the next one;
  - with `idle_minutes` set and a fake clock past the deadline, the server exits and removes the
    socket;
  - a second `serve()` while the lock is held returns immediately.
- [ ] Run the tests. They fail with an import error.
- [ ] Implement `protocol.py` and `server.py`.
- [ ] Run the tests. They pass. Commit `feat(cli): add the daemon socket protocol and server`.

### Task 2: Client, CLI and lifecycle

**Files:**
- Create: `src/macecho/cli.py`
- Test: `tests/test_agent_cli.py`

**Interfaces it consumes:** `protocol.*`.

**Interfaces it produces:**
- `cli.main(argv=None) -> int`.
- `cli.request(obj, spawn=_spawn_daemon, wait=600) -> dict`. Connects, or removes a stale socket
  and spawns the daemon, then retries once after a crash.
- `cli.exit_code(reply) -> int`.

**Steps:**
- [ ] **Tests:**
  - `listen` with an ok reply prints only the text and returns 0;
  - `--json` prints `{"text", "seconds"}`;
  - a timeout reply returns 2 with empty stdout;
  - `mic_denied` returns 3 and prints the System Settings path to stderr;
  - an empty `say` returns 1 without any request being sent;
  - a stale socket file is removed and spawn is called;
  - a reset connection is retried exactly once;
  - `status` with no daemon prints `{"running": false}` and does not spawn.
- [ ] Run them; they fail. Implement `cli.py`; run them; they pass.
- [ ] Commit `feat(cli): add the macecho client with auto-start and exit codes`.

### Task 3: Engine, listening

**Files:**
- Create: `src/macecho/daemon/engine.py`
- Test: `tests/test_agent_cli.py`

**Interfaces it produces:**
- `engine.capture(frames: Iterable[np.ndarray], vad, transcribe, timeout, clock) -> str | None`.
  Feeds 512-sample frames to the VAD until it reports `SPEECH_END`, then returns the transcript.
  Empty transcripts are skipped. Returns `None` on timeout. Raises `EngineError("mic_denied")` if
  the first 1.0 s of frames is all exact zeros.
- `Engine(repo, voice_name)` with `listen`, `say` and `status`.

**Steps:**
- [ ] **Tests**, using the real `VadProcessor` and `src/macecho/vad/en.wav` padded with 1 s of
  silence on each side:
  - one transcript is returned;
  - a transcriber returning `""` once, then `"hi"`, yields `"hi"`;
  - pure silence with low-level noise yields `None` once the timeout passes;
  - all-zero frames raise `mic_denied`.
- [ ] Run them; they fail. Implement `capture` and `Engine.listen`; run them; they pass.
- [ ] Commit `feat(cli): listen for one utterance and transcribe it`.

### Task 4: Voices and speaking

**Files:**
- Create: `src/macecho/daemon/voice.py`
- Create: `src/macecho/daemon/__main__.py`
- Modify: `src/macecho/daemon/engine.py`

**Interfaces it produces:**
- `voice.speakable(text) -> str`. Removes markdown symbols, code fences and URLs.
- `SystemVoice().speak(text)`.
- `LfmVoice(speech_model).speak(text)`. Uses the `Perform TTS.` prompt with
  `generation_mode="audio"` and streams to an `AudioPlayer` at 24 kHz.
- `voice.make(name, speech_model)`.

**Steps:**
- [ ] **Tests:** `speakable("`npm test` **passed** see https://x.y")` returns `"npm test passed"`;
  `SystemVoice` calls `say` with the cleaned text (subprocess faked); `make("system")` and
  `make("lfm3")` choose correctly.
- [ ] Run them; they fail. Implement; run them; they pass.
- [ ] Add `python -m macecho.daemon`: build the `Engine` and call `serve()`.
- [ ] Commit `feat(cli): speak with LFM3's voice or the system voice`.

### Task 5: Agent guide, skill auto-install, `guide`, `install-skill`

**Files:**
- Create: `src/macecho/agent_guide.md`
- Create: `src/macecho/skill/SKILL.md`
- Modify: `src/macecho/cli.py`
- Modify: `AGENTS.md` (create it)

**Steps:**
- [ ] **Tests**, with a temporary `HOME`:
  - auto-install writes the skill when `~/.claude` exists and the skill is absent;
  - nothing happens when `~/.claude` is missing, when the skill is present, or when
    `MACECHO_NO_SKILL=1`;
  - `install-skill --force` overwrites;
  - `guide` prints the guide text.
- [ ] Run them; they fail. Implement; run them; they pass.
- [ ] Commit `feat(cli): ship an agent guide and auto-install the Claude Code skill`.

### Task 6: Packaging, documentation and end-to-end check

**Files:**
- Create: `pyproject.toml`
- Delete: `setup.py`
- Modify: `README.md`
- Modify: `scripts/build_app.sh`, only if it relies on `setup.py`

**Steps:**
- [ ] Write `pyproject.toml`:
  - `[project.scripts] macecho = "macecho.cli:main"`;
  - core dependencies;
  - optional extra `app = [rumps, pyobjc-framework-WebKit]`;
  - package data: `*.onnx`, `*.md`, `ui/index.html`.
- [ ] `uv tool install --force .` into a temporary tool directory, then run `macecho --help`
  and `macecho status`.
- [ ] **End to end, outside the sandbox:**
  - `say -o q.aiff "what time is it"`;
  - run `Engine.capture` on that file through the real model; the transcript should match;
  - `macecho say "Tests passed."` should be heard;
  - `macecho stop`.
- [ ] Add a "Use it from an AI agent" section to `README.md`.
- [ ] Run the full test suite. Commit
  `build: install macecho as a uv tool with a pyproject`.
