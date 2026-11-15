# MacEcho drive (ACP): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** `macecho drive <agent>` runs a coding agent by voice: "Friday, *task*" starts work, a
short summary is spoken when it ends, permission requests are announced, and "stop" halts it.

**Architecture:**
- **`acp.py`:** a standard-library ACP client. It spawns the agent, runs the handshake and turns
  the agent's messages into events on a queue.
- **`drive.py`:** a single-threaded loop over one queue fed by the agent, the voice and the
  keyboard. The voice and the agent are passed in, so tests use fakes.
- **Daemon:** unchanged apart from one addition: a `listen` ends early when its client hangs up.

**Tech stack:** Python 3.12, stdlib `subprocess`, `threading`, `queue`, `json`, `select`,
`socket`, `argparse`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-14-macecho-drive-acp-design.md`

## Global constraints

- Python `>=3.12,<3.14`. No new dependencies.
- `cli.py`, `acp.py` and `drive.py` must not import MLX, numpy or sounddevice. `drive.py` may
  import `macecho.daemon.voice` (it uses only `re` and `subprocess`).
- ACP protocol version `1`. Messages are UTF-8 JSON-RPC 2.0, one per line, with no embedded
  newlines.
- The wake word is "Friday".
- Spoken strings, verbatim:
  - "Ready. Say Friday, then your task."
  - "On it."
  - "Yes?"
  - "Okay, talk soon."
  - "*Name* needs your attention."
  - "Stopped."
  - "Stopped. I had to restart *Name*, so it has forgotten this conversation."
  - "*Name* hit its limit before finishing."
  - "*Name* declined that request."
  - "*Name* hit an error."
  - "*Name* has stopped unexpectedly."
  - "*Name* has finished." (a turn that ended with nothing speakable)
- Timings: stop grace 10 s; the window after a bare "Friday" 10 s; handshake timeout 60 s.
- Exit codes of `drive`: `0` ended by the user, `1` the agent could not start or died, `3`
  microphone access is off.
- Run tests with `PYTHONPATH=src .venv/bin/python -m pytest tests/`. Tests that bind Unix sockets
  (`tests/test_agent_cli.py`) fail inside the Claude Code sandbox and must run with it off. The
  tests in `tests/test_drive.py` use only pipes and run inside it.
- Commit after each task, with the message given in the task.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `src/macecho/daemon/engine.py` | modify | `capture()` and `Engine.listen()` take an `abandoned` check. |
| `src/macecho/daemon/server.py` | modify | Passes a check that polls the client's socket. |
| `src/macecho/acp.py` | create | The ACP client. Knows nothing about voice. |
| `src/macecho/drive.py` | create | The words (wake word, stop, goodbye, summary), the loop, and the voice that talks to the daemon. |
| `src/macecho/cli.py` | modify | The `drive` subcommand. |
| `tests/fake_acp_agent.py` | create | A scripted ACP agent for tests. |
| `tests/test_drive.py` | create | Tests for `acp.py` and `drive.py` that need no socket. |
| `tests/test_agent_cli.py` | modify | Tests for the abandonable listen and the daemon voice. |
| `README.md` | modify | A "Drive a coding agent by voice" section and the layout list. |

## Review focus

Conditions the spec implies but the obvious tests miss, each pinned by a test in its owning task:

1. **The agent prints a line that is not JSON on stdout**, such as a start-up banner. The line is
   logged and skipped; the session carries on. *Test: Task 2, the `reply` scenario prints one
   before answering `initialize`.*
2. **The agent's command is not installed, or it exits during the handshake.** `start()` raises
   `AcpError` with a message naming the command, at once, not after the 60 s timeout. *Test:
   Task 2.*
3. **`session/prompt` fails with a JSON-RPC error**, for example when the harness is not logged
   in. The loop says "*Name* hit an error.", prints the message and returns to Idle. *Test:
   Tasks 2 and 4.*
4. **Two permission requests arrive before the first is answered.** Both are answered, in order;
   neither is lost. *Test: Task 4.*
5. **The transcriber punctuates the wake word**, as in "Friday!" or "Friday?", or the user says
   "Friday" again inside the 10 s window ("Friday" … "Friday, run the tests"). The first is a
   bare wake word; the second sends "run the tests", not "Friday, run the tests". *Test: Tasks 3
   and 4.*

---

- [ ] **Before Task 1:** commit the spec and this plan: `docs: add the drive spec and plan`.

### Task 1: A listen that ends when its client hangs up

**Files:**
- Modify: `src/macecho/daemon/engine.py` (`capture`, `Engine.listen`)
- Modify: `src/macecho/daemon/server.py` (`handle`, `_serve_one`, new `_hung_up`)
- Test: `tests/test_agent_cli.py`

**Interfaces it produces:**
- `engine.capture(frames, vad, transcribe, timeout, abandoned=lambda: False) -> str | None`.
  Calls `abandoned()` once per frame; when it returns true, raises
  `EngineError("abandoned", "the caller hung up")`.
- `Engine.listen(timeout, abandoned=lambda: False) -> dict`.
- `server.handle(engine, request, abandoned=lambda: False) -> dict`. Passes `abandoned` to
  `engine.listen` for the `listen` and `converse` ops.
- `server._hung_up(conn) -> bool`: true when `select.select([conn], [], [], 0)` reports the
  socket readable and `conn.recv(1, socket.MSG_PEEK)` returns `b""`.

**Steps:**
- [ ] **Tests.** Change `FakeEngine.listen` to `listen(self, timeout, abandoned=lambda: False)`.
  Add:
  - `capture(frames(room(5)), vad(), lambda _: "never", timeout=60, abandoned=...)`, where the
    check returns true from its 10th call, raises `EngineError` with kind `"abandoned"`;
  - with the real server and an engine whose `listen` loops `while not abandoned():
    time.sleep(0.01)` and then raises `EngineError("abandoned")`: a client sends a `listen` and
    closes; `ask({"op": "say", "text": "next"})` then returns `{"ok": True}` within 2 s;
  - the existing tests still pass unchanged.
- [ ] Run them with the sandbox off. The two new ones fail.
- [ ] Implement. In `_serve_one`, call `handle(engine, request, lambda: _hung_up(conn))`.
- [ ] Run all of `tests/test_agent_cli.py` with the sandbox off. They pass.
- [ ] Commit `feat(daemon): end a listen when its client hangs up`.

### Task 2: The ACP client and the fake agent

**Files:**
- Create: `src/macecho/acp.py`
- Create: `tests/fake_acp_agent.py`
- Test: `tests/test_drive.py`

**Interfaces it produces:**

```python
class AcpError(Exception): ...

class Agent:
    def __init__(self, command: list[str], cwd: str, events: queue.Queue, log=print): ...
    name: str | None                      # agentInfo.title or .name, after start()
    def start(self) -> None               # spawn, initialize, session/new; raises AcpError
    def prompt(self, text: str) -> None   # returns at once; the turn's end is a "done" event
    def cancel(self) -> None              # session/cancel notification
    def answer(self, request_id, option_id: str | None) -> None   # None answers "cancelled"
    def close(self, grace: float = 2.0) -> None   # close stdin; terminate, then kill, after grace
```

Events put on `events`, as tuples:

| Event | When |
|---|---|
| `("text", str)` | A `session/update` with `sessionUpdate: "agent_message_chunk"` and a text block. |
| `("tool", title)` | A `session/update` with `sessionUpdate: "tool_call"`. |
| `("permission", request_id, title, options)` | A `session/request_permission`. `title` is `toolCall.title`, or `toolCall.toolCallId` if there is none. `options` is the list of `{"optionId", "name", "kind"}` dicts as received. |
| `("done", stop_reason)` | The reply to `session/prompt`. `stop_reason` is the `stopReason`, or `"error"` when the reply is a JSON-RPC error (its message goes to `log`). |
| `("exit", returncode)` | The child's stdout closed. Not sent after `close()`. |

**Wire details:**
- Spawn with `stdin=PIPE, stdout=PIPE`, stderr inherited, `cwd=cwd`, text mode, UTF-8, line
  buffered. A `FileNotFoundError` becomes `AcpError("<command> is not installed or not on PATH")`.
- `initialize` params: `{"protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name":
  "macecho", "title": "MacEcho", "version": <package version, or "0">}}`. A reply whose
  `protocolVersion` is not `1` raises `AcpError`.
- `session/new` params: `{"cwd": <absolute cwd>, "mcpServers": []}`. Keep `sessionId`. An error
  reply raises `AcpError` with the agent's message and the hint "log in with the agent's own
  command first".
- `session/prompt` params: `{"sessionId", "prompt": [{"type": "text", "text": text}]}`.
- Permission reply: `{"outcome": {"outcome": "selected", "optionId": id}}` or
  `{"outcome": {"outcome": "cancelled"}}`.
- Any other request from the agent gets the error `{"code": -32601, "message": "Method not
  found"}`. Other notifications are ignored.
- One reader thread parses stdout. A line that is not JSON is passed to `log` and skipped. A lock
  guards writes to stdin; a `BrokenPipeError` on write is swallowed (the `exit` event follows).
- When stdout closes during the handshake, the waiting `start()` raises `AcpError` at once.

**The fake agent** is run as `python tests/fake_acp_agent.py <scenario>`. It answers `initialize`
(`protocolVersion: 1`, `agentInfo: {"name": "fake", "title": "Fake"}`) and `session/new`
(`sessionId: "s1"`), then on `session/prompt`:

| Scenario | Behaviour |
|---|---|
| `reply` | Prints the line `starting up` (not JSON) before answering `initialize`. On a prompt: a `tool_call` titled "Read file", the chunks `"Working on it.\n\n"` and `"SPOKEN: All done. Two tests pass."`, then `end_turn`. |
| `permission` | Sends `session/request_permission` (id 100, title "Run rm -rf build", options `allow`/"Allow"/`allow_once` and `reject`/"Reject"/`reject_once`). On `selected`: the chunk `"SPOKEN: You chose <optionId>."`, then `end_turn`. On `cancelled`: waits for `session/cancel`, then `cancelled`. |
| `two-permissions` | As `permission`, but sends ids 100 and 101 before reading any answer, and ends the turn after both are answered with `"SPOKEN: You chose <first>, <second>."`. |
| `slow` | Sends the chunk `"Starting."`, then waits. On `session/cancel`: `cancelled`. |
| `deaf` | Sends the chunk `"Starting."`, then ignores everything, including SIGTERM. |
| `asks` | Sends the request `fs/read_text_file` (id 7), then the chunk `"SPOKEN: error <code>."` with the error code it got back, then `end_turn`. |
| `fails` | Answers the prompt with the JSON-RPC error `{"code": -32000, "message": "not logged in"}`. |
| `crash` | Exits with code 3 on the prompt. |
| `old` | Answers `initialize` with `protocolVersion: 99`. |
| `dies` | Exits with code 1 on `initialize`, without answering. |

**Steps:**
- [ ] Write `tests/fake_acp_agent.py`.
- [ ] **Tests**, each starting `Agent([sys.executable, FAKE, scenario], cwd=tmp_path, events)`:
  - `reply`: `start()` sets `name == "Fake"`; after `prompt("hi")` the events are
    `("tool", "Read file")`, `("text", "Working on it.\n\n")`,
    `("text", "SPOKEN: All done. Two tests pass.")`, `("done", "end_turn")`; the junk line
    reached `log`;
  - `permission`, answered `"allow"`: the text event is `"SPOKEN: You chose allow."`;
  - `permission`, answered `None`, then `cancel()`: `("done", "cancelled")`;
  - `slow`, then `cancel()`: `("done", "cancelled")`;
  - `asks`: the text event is `"SPOKEN: error -32601."`;
  - `fails`: `("done", "error")`, and `log` received "not logged in";
  - `crash`: `("exit", 3)`;
  - `old`: `start()` raises `AcpError`;
  - `dies`: `start()` raises `AcpError` in under 5 s;
  - a command that does not exist: `start()` raises `AcpError` naming it;
  - `deaf`: `close(grace=0.5)` returns in under 3 s, the process is gone, and no `exit` event is
    queued.
- [ ] Run them. They fail with an import error.
- [ ] Implement `acp.py`.
- [ ] Run them inside the sandbox. They pass.
- [ ] Commit `feat(drive): add a minimal ACP client`.

### Task 3: The words: wake word, stop, goodbye and summary

**Files:**
- Create: `src/macecho/drive.py` (these functions only)
- Test: `tests/test_drive.py`

**Interfaces it produces:**
- `WAKE_WORDS: tuple[str, ...]`: accepted spellings of the wake word, lower case. Starts as
  `("friday", "fri day")`.
- `after_wake_word(text: str) -> str | None`: the rest of the sentence when `text` starts with a
  wake word (as a whole word, after an optional "hey", "ok" or "okay"), with leading punctuation
  and spaces removed and its case kept; `""` when nothing follows; `None` otherwise.
- `is_stop(text: str) -> bool`: true when, after lower-casing, removing punctuation, a leading
  wake word and leading "please", "ok", "okay" or "hey", the utterance has at most three words and
  the first is "stop", "cancel", "abort" or "halt".
- `is_goodbye(task: str) -> bool`: true when the normalised text is "goodbye", "good bye", "bye",
  "bye bye" or "stop listening".
- `spoken_summary(text: str) -> str`: the text after the last `SPOKEN:` (any case), stripped. If
  there is no marker, or nothing follows it, the first two sentences of the last paragraph that
  has speakable words, using `voice.speakable` and `voice.sentences`. `""` when nothing is
  speakable.
- `PROMPT_PREFIX: str`, ending in `Task: ` with one trailing space, so the prompt is
  `PROMPT_PREFIX + task`:

  ```
  The user is talking to you by voice through MacEcho and is not looking at the screen. Do the
  task below. End your reply with one final line that starts with "SPOKEN:" and says, in one or
  two short plain sentences with no code, file paths or markdown, what you did and whether you
  need a decision from the user.

  Task:
  ```

**Steps:**
- [ ] **Tests**, as tables:

  | `after_wake_word` input | Result |
  |---|---|
  | `"Friday, fix the failing test."` | `"fix the failing test."` |
  | `"friday fix it"` | `"fix it"` |
  | `"Hey Friday, run the tests"` | `"run the tests"` |
  | `"Fri day, run the tests"` | `"run the tests"` |
  | `"Friday."`, `"Friday!"`, `"Friday?"`, `"friday"` | `""` |
  | `"See you on Friday"`, `"It's Friday"`, `"Fridaynight plans"`, `""` | `None` |

  | `is_stop` input | Result |
  |---|---|
  | `"Stop."`, `"stop it"`, `"STOP!"`, `"Friday, stop"`, `"Stop Claude"`, `"cancel that"`, `"please stop"`, `"stop stop stop"` | `True` |
  | `"don't stop"`, `"we should stop by the store later"`, `"the stop sign on the corner"`, `""`, `"fix the test"` | `False` |

  | `is_goodbye` input | Result |
  |---|---|
  | `"goodbye"`, `"Good bye."`, `"Bye!"`, `"stop listening"` | `True` |
  | `"stop"`, `"say goodbye to the old API"` | `False` |

  | `spoken_summary` input | Result |
  |---|---|
  | `"Edited two files.\n\nSPOKEN: I fixed the login test. Nothing needs your decision."` | `"I fixed the login test. Nothing needs your decision."` |
  | `"SPOKEN: first.\n\nMore work.\n\nspoken: second."` | `"second."` |
  | `"I looked around.\n\nFixed the bug in auth. Tests pass now. Also cleaned imports."` | `"Fixed the bug in auth. Tests pass now."` |
  | `"Fixed it. All good.\n\nSPOKEN:"` | `"Fixed it. All good."` |
  | `"```\nonly code\n```"`, `""` | `""` |

- [ ] Run them; they fail. Implement; run them; they pass inside the sandbox.
- [ ] Commit `feat(drive): recognise the wake word, stop and the spoken summary`.

### Task 4: The loop

**Files:**
- Modify: `src/macecho/drive.py`
- Test: `tests/test_drive.py`

**Interfaces it consumes:** `acp.Agent`, `acp.AcpError` (Task 2); the words (Task 3).

**Interfaces it produces:**

```python
STOP_GRACE = 10.0   # seconds to wait for a cancelled turn to end before killing the agent
WAKE_WINDOW = 10.0  # seconds after "Yes?" in which the next utterance needs no wake word

def run(make_agent, voice, events: queue.Queue, fallback_name: str,
        clock=time.monotonic, log=print, poll: float = 0.1) -> int
```

- `make_agent()` returns an `Agent` that has not been started; `run` calls `start()`. It is called
  once at the beginning and again after a forced restart, with the next task.
- `voice` has `say(text) -> None`, which blocks until the speech has finished, and `close()`.
- `run` returns the exit code: `0`, `1` or `3`.
- Events it reads, besides the agent's: `("heard", text)`, `("key", line)`, `("mic_denied",)`.

**Behaviour:** the state table in the spec, plus:
- *Name* is `agent.name or fallback_name`.
- Start the agent before anything is spoken. If `start()` raises `AcpError`, log the message and
  return `1` without speaking.
- On "Friday, *task*": send `PROMPT_PREFIX + task`, then say "On it."
- A `tool` event between two `text` events inserts a blank line between them in the collected
  text, so the last paragraph is the agent's last message.
- Attention prints the title and the options numbered from 1 (`1) Allow   2) Reject`) and the
  line `Type a number, then Return.` A `key` event with a valid number answers that option. Any
  other key reprints the options. Keys in other states are ignored.
- A `permission` event while Stopping or Idle is answered `None` at once.
- A `done` event in Stopping is always followed by "Stopped.", whatever its stop reason.
- Deadlines (the stop grace and the wake window) are checked against `clock()` each time the
  queue wait returns, so a test advances a fake clock.
- `mic_denied`: log the help text from `cli.MIC_HELP`, close the agent, return `3`.
- `KeyboardInterrupt`: close the agent and the voice, return `0`.

**Steps:**
- [ ] **Tests.** A `FakeVoice` records what is said. Each test puts `heard` and `key` events on
  the queue itself and uses the fake agent from Task 2, with `poll=0.02`:
  - `reply`: heard "Friday, add a test" then "Friday, goodbye". Said, in order: "Ready. Say
    Friday, then your task.", "On it.", "All done. Two tests pass.", "Okay, talk soon." Returns
    `0`. The agent's `prompt` was called with text ending in "Task: add a test" (the test's
    `make_agent` wraps `prompt` to record it);
  - speech without the wake word in Idle ("add a test") sends nothing and says nothing;
  - "Friday." is answered "Yes?"; "add a test" inside the window is sent as the task; with the
    fake clock moved 11 s on first, it is ignored;
  - "Friday." then "Friday, add a test" sends "add a test";
  - `permission`: "Fake needs your attention." is said; key "9" reprints; key "1" answers
    `allow`; the summary "You chose allow." is spoken;
  - `two-permissions`: keys "1" then "2" answer `allow` then `reject`; "Fake needs your
    attention." is said once per request;
  - `permission`, then heard "stop": the request is answered cancelled, "Stopped." is said, and
    the loop is Idle again (a following "Friday, goodbye" returns `0`);
  - `slow`, then heard "keep going" (ignored), then heard "Stop.": "Stopped." is said;
  - `deaf`, then heard "stop", then the fake clock moved 11 s on: "Stopped. I had to restart
    Fake, so it has forgotten this conversation." is said, and `make_agent` is called a second
    time when the next task arrives;
  - `fails`: "Fake hit an error." is said and the loop stays Idle;
  - `crash`: "Fake has stopped unexpectedly." is said; returns `1`;
  - `make_agent` returning an agent whose command does not exist: returns `1`, nothing said;
  - a `mic_denied` event: returns `3`.
- [ ] Run them; they fail. Implement `run`; run them; they pass inside the sandbox.
- [ ] Commit `feat(drive): add the voice loop over an ACP agent`.

### Task 5: The daemon voice, the `drive` command and the documentation

**Files:**
- Modify: `src/macecho/drive.py` (`DaemonVoice`, `main`)
- Modify: `src/macecho/cli.py`
- Modify: `README.md`
- Test: `tests/test_agent_cli.py`

**Interfaces it consumes:** `cli.request`, `cli._connect`, `cli._start_daemon`,
`cli._spawn_daemon`; `protocol.send`, `protocol.recv`; `run` (Task 4).

**Interfaces it produces:**
- `drive.DaemonVoice(events)`, with `say(text)` and `close()`.
  - A listener thread keeps one `{"op": "listen", "timeout": 3600}` request open whenever the
    voice is not speaking. A reply with text becomes `("heard", text)`. A `timeout` reply opens
    the next listen. A `mic_denied` reply becomes `("mic_denied",)` and ends the thread.
  - `say` ends the open listen with `sock.shutdown(socket.SHUT_RDWR)`, sends the `say` with
    `cli.request`, waits for its reply, then lets the listener open a new listen. `shutdown` is
    required: closing a socket from another thread does not wake a `recv` blocked on it.
- `drive.AGENTS = {"claude": ["claude-agent-acp"], "gemini": ["gemini", "--acp"],
  "opencode": ["opencode", "acp"]}`.
- `drive.main(command: list[str], fallback_name: str) -> int`: makes sure the daemon is up with
  `cli.request({"op": "status"})`, starts a thread that turns stdin lines into `("key", line)`
  events, and calls `run` with `cwd=os.getcwd()`.
- `macecho drive <name>` and `macecho drive -- <command> [...]` in `cli.py`. An unknown name
  prints the known names and the `--` form to stderr and returns `1`. The fallback name is the
  given name capitalised, or the base name of the command.

**Steps:**
- [ ] **Tests** (sandbox off):
  - with the real server and an engine whose `listen` blocks until abandoned or until the test
    releases a sentence: `DaemonVoice.say("hello")` returns within 2 s and the engine recorded
    `("say", "hello")`; a sentence released afterwards arrives as `("heard", "the sentence")`;
  - `cli.main(["drive", "nonsense"])` returns `1` and stderr names `claude`, `gemini` and
    `opencode`;
  - `cli.main(["drive", "--", sys.executable, FAKE, "dies"])` returns `1`.
- [ ] Run them; they fail. Implement; run all tests with the sandbox off; they pass.
- [ ] **README:** add "Drive a coding agent by voice" after "Quick start", covering: the command,
  the wake word, "stop", the permission key press, the adapters each harness needs, and that the
  mic stays on for the whole session. Add `acp.py` and `drive.py` to the layout list and the
  `drive` rows to the command reference.
- [ ] Commit `feat(cli): add macecho drive`.

### Task 6: End-to-end check with real harnesses

Needs the sandbox off, the GPU, audio, headphones and a logged-in harness. The user speaks.

**Steps:**
- [ ] From a scratch git repository, run `macecho drive claude`. Check each of:
  - "Ready. Say Friday, then your task." is heard;
  - a sentence without "Friday" is logged as ignored;
  - "Friday, create a file called hello.txt that says hi" is heard as "On it.", and ends with a
    spoken summary of one or two sentences;
  - a task that triggers a permission request is announced, and the key press works;
  - "stop" during a long task ends the turn. Time it from the end of the word, three times.
- [ ] Repeat the start-to-summary check with `gemini` and `opencode`. Confirm `opencode acp` is
  the right command, or correct `AGENTS`.
- [ ] Read the terminal log for how the transcriber wrote "Friday" and "stop". Add any missed
  spellings to `WAKE_WORDS`, with a test row each.
- [ ] Record the results in the spec: the measured stop delay, each harness's result, and any
  spellings added. Set the spec's status to implemented.
- [ ] Commit `docs: record the drive end-to-end results`.
