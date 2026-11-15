# MacEcho drive: run a coding agent by voice over ACP

Date: 2026-11-14 · Status: design approved in conversation, awaiting spec review

## Goal

Let the user run a coding agent without looking at the screen. The user says "Friday" and then a
task. MacEcho sends it to the agent, says that work has started, and stays quiet while the agent works. When the
agent finishes, MacEcho speaks a short summary. When the agent needs a decision, MacEcho says so.
Saying "stop" halts the agent.

MacEcho talks to the agent over the Agent Client Protocol (ACP), so the agent can be Claude Code,
Gemini CLI, OpenCode, Codex or Pi without changing MacEcho.

**Success:** from a project folder, `macecho drive claude` runs a task spoken as "Friday, ..." end
to end. Speech that does not start with "Friday" is never sent to the agent. The user hears
"On it", then one or two sentences when the work is done. Saying "stop" during the work ends
the turn within a few seconds.

### Out of scope for v1

- Starting `drive` by voice, and running it inside the Mac app. v1 is a terminal command.
- Approving a permission request by voice. A misheard "yes" would run a command.
- Adding to a task while the agent works. Speech other than "stop" is ignored until the turn ends.
- ACP client file-system and terminal methods. MacEcho advertises neither, so the agent uses its
  own.
- Authenticating a harness. The user logs in with the harness itself first.
- Resuming an earlier session (`session/load`).

## Command surface

```
macecho drive <name>              # claude | gemini | opencode
macecho drive -- <command> [...]  # any other ACP agent, e.g. a Codex or Pi adapter
```

| Name | Command it spawns | Checked on this Mac (2026-11-14) |
|---|---|---|
| `claude` | `claude-agent-acp` | Installed, version 0.58.1. Not yet run. |
| `gemini` | `gemini --acp` | Flag present in `gemini --help`. Not yet run. |
| `opencode` | `opencode acp` | Not confirmed: the sandbox blocked `opencode` from starting. |

The session's working directory is the directory `drive` is run from. `drive` runs until the user
says "goodbye" or presses Ctrl-C. It prints a text log of the session to the terminal: what it
heard, the agent's messages, tool call titles and permission requests.

Exit codes: `0` ended by the user, `1` the agent could not be started or died, `3` microphone
access is off (as for `listen`).

## Architecture

```
            ┌─ voice ──── Unix socket ────▶ macecho daemon (unchanged model, VAD, mic, voice)
macecho drive
            └─ ACP ── stdio, JSON-RPC ───▶ harness (claude-agent-acp, gemini --acp, ...)
```

`drive` is the ACP client. It starts the harness as a child process and is the harness's only
user interface. It uses the existing daemon to listen and speak, so the speech engine is untouched
apart from one addition: a listen can be abandoned (see Listening while the agent works).

### Units

- **`macecho/acp.py`: a minimal ACP client.** Standard library only. It spawns the agent, runs
  the handshake, and turns the agent's messages into events on a queue. It knows nothing about
  voice.
  - `start(command, cwd)`: spawn, `initialize`, `session/new`.
  - `prompt(text)`: send `session/prompt`; the result arrives later as a `done` event.
  - `cancel()`: send `session/cancel`.
  - `answer(request_id, option_id | None)`: reply to a permission request; `None` means cancelled.
  - `close(grace)`: end the child; kill it if it has not exited after `grace` seconds.
  - Events: `text` (a message chunk), `tool` (a tool call title, for the log), `permission`
    (request id, title, options), `done` (stop reason), `exit` (the child ended).
- **`macecho/drive.py`: the loop.** A single-threaded state machine that reads one queue fed by
  three sources: the agent's events, the user's utterances and the keyboard. It decides what to
  say and what to send. The voice and the agent are passed in, so tests use fakes.
- **`macecho/daemon/server.py` and `engine.py`: abandonable listen.** See below.
- **`macecho/cli.py`: the `drive` subcommand.** Parses the name or command and calls `drive`.

### ACP messages used

Checked against agentclientprotocol.com on 2026-11-14. Messages are UTF-8 JSON-RPC 2.0, one per
line, on the agent's stdin and stdout. The agent's stderr is log text and goes to the terminal log.

| Direction | Message | Use |
|---|---|---|
| to agent | `initialize` | `protocolVersion: 1`, `clientInfo`, no client capabilities. |
| to agent | `session/new` | `cwd` (absolute) and `mcpServers: []`. Returns `sessionId`. |
| to agent | `session/prompt` | The task as one text block. Returns `stopReason` when the turn ends. |
| from agent | `session/update` | `agent_message_chunk` is collected; `tool_call` is logged; other kinds are ignored. |
| from agent | `session/request_permission` | Answered with the chosen `optionId`, or `cancelled`. |
| to agent | `session/cancel` | A notification. The agent must then end the turn with `stopReason: "cancelled"`. |

If the agent answers `initialize` with a protocol version other than 1, `drive` says it cannot use
this agent and exits `1`. Any other request from the agent gets a JSON-RPC "method not found"
error.

## The loop

| State | Event | What MacEcho does | Next state |
|---|---|---|---|
| start | The agent and the voice are up | Says "Ready. Say Friday, then your task." | Idle |
| Idle | "Friday, *task*" | Says "On it." Sends the task, without the wake word. | Working |
| Idle | "Friday" alone | Says "Yes?" The next utterance within 10 s is the task and needs no wake word. | Idle |
| Idle | "Friday, goodbye" | Says "Okay, talk soon." Closes the agent. | exit |
| Idle | Anything else | Logs that it was ignored. Nothing is sent. | Idle |
| Working | Agent text or tool call | Logs it. Keeps the text. | Working |
| Working | Permission request | Says "*Name* needs your attention." Prints the request and its options. | Attention |
| Working | The user says "stop" | Sends `session/cancel`. | Stopping |
| Working | The user says anything else | Logs it and ignores it. | Working |
| Working | Turn ends | Speaks the summary or the reason (below). | Idle |
| Attention | A key press chooses an option | Answers the request. If more requests are waiting, shows the next. | Working, or Attention |
| Attention | The user says "stop" | Answers the request `cancelled`, then sends `session/cancel`. | Stopping |
| Stopping | Turn ends | Says "Stopped." | Idle |
| Stopping | 10 s pass with no end | Kills the agent. Says "Stopped. I had to restart *Name*, so it has forgotten this conversation." A new agent starts with the next task. | Idle |
| any | The agent process exits | Says "*Name* has stopped unexpectedly." | exit `1` |

*Name* is the agent's `agentInfo.title` or `name` from `initialize`, or the name given on the
command line.

A permission request that arrives while Stopping or Idle is answered `cancelled` at once. Requests
that arrive while another is waiting are queued and shown one at a time.

### The wake word

The wake word is "Friday". There is no separate wake-word model: every utterance is transcribed
on the Mac, as now, and one function, `after_wake_word(text) -> str | None`, returns the rest of
the sentence if it starts with the wake word, or `None`. Nothing reaches the agent, or leaves the
Mac, unless it does.

- **Loose matching.** The transcriber may write "Fri day" or "Friday's", so the function accepts a
  short list of spellings, held in one constant. The list is tuned from what the log shows the
  model actually heard.
- **Pausing after the name.** People pause after a wake word, and 0.6 s of silence ends an
  utterance. So "Friday" alone is answered with "Yes?", and the next utterance is the task.
- **Not needed for "stop".** While the agent works, a bare "stop" counts, and so does "Friday,
  stop". Stopping is the safe direction, so it should not depend on the wake word being heard
  correctly.
- **Needed for everything else,** including "goodbye".

### What "stop" does

`session/cancel` asks the harness to abort its model request and any running tool. Changes already
made to files stay; stop is not undo. If the harness does not end the turn within 10 s, MacEcho
kills the process, which always works but loses the session's memory.

Expected delay from the end of the word to the cancel being sent: about 1.3 s. That is 0.6 s for
the VAD to decide the user has finished, plus transcription (0.71 s was measured for a full
sentence; a single word should be faster). This is an estimate, not a measurement.

**Recognising "stop".** One function, `is_stop(text) -> bool`, decides. Stopping by mistake costs a
restart of the task; missing a stop lets the agent carry on. So it should lean towards stopping:
short utterances that are a stop word or phrase ("stop", "cancel", "stop it", "stop Claude") match;
long sentences that merely contain the word do not. The exact rule is settled in implementation,
with a table of examples as its test.

### What is spoken when a turn ends

| Stop reason | Spoken |
|---|---|
| `end_turn` | The agent's spoken summary. |
| `cancelled` | "Stopped." |
| `max_tokens`, `max_turn_requests` | "*Name* hit its limit before finishing." |
| `refusal` | "*Name* declined that request." |
| The prompt fails with an error | "*Name* hit an error." The error message is printed. |

**The summary.** Every prompt is sent with a short instruction placed before the user's words:
finish with a line starting `SPOKEN:` that says in one or two plain sentences what was done, and
whether the user must decide anything. MacEcho speaks the text after the last `SPOKEN:`. If the
agent leaves the line out, MacEcho speaks the first two sentences of the agent's last paragraph.
The daemon already removes markdown, code and links before speaking.

The agent writes the summary because it knows what it did. LFM3 only voices it.

### Listening while the agent works

Today the daemon serves one request at a time, and a `listen` cannot be ended early. `drive` must
listen for "stop" during the work and also speak the moment a permission request arrives.

So a `listen` becomes abandonable: if the client closes its connection, the daemon stops listening
at the next audio frame (32 ms) and takes the next request. `capture()` gains an `abandoned()`
check, and the server passes one that polls the client's socket. `drive` keeps one `listen` open
whenever it is not speaking, and closes it before each `say`.

Because speaking and listening never overlap, MacEcho does not hear its own voice.

## Testing

**Automated tests** need no GPU, mic or real harness, and run in the sandbox:

- **A fake ACP agent** (`tests/fake_acp_agent.py`): a script that speaks ACP on stdio and follows
  a scenario given on its command line: reply and end the turn; ask for permission; honour a
  cancel; ignore a cancel.
- **`acp.py` against the fake agent:** the handshake, a prompt that streams text and ends, a
  permission round trip in both outcomes, a cancel that ends with `cancelled`, and `close` killing
  an agent that ignores it.
- **The loop** with a fake voice and the fake agent: every row of the state table, including the
  10 s kill with an injected clock.
- **`after_wake_word`, `is_stop` and the summary picker:** tables of examples, including "Friday"
  alone, misspellings, and a sentence that only contains the word ("see you on Friday").
- **Abandonable listen:** with the real server and a fake engine, a client that hangs up
  mid-listen frees the daemon for the next request.

**Manual checks** need the sandbox off, the GPU, audio and a logged-in harness:

- With `macecho drive claude`: speak a task and hear the summary; speak a task that triggers a
  permission request; say "stop" mid-task and time it.
- The same start-to-summary check with `gemini` and `opencode`.

## Risks

- **False wake-ups.** A sentence that happens to start with "Friday" ("Friday is the deadline")
  becomes a task. The spoken "On it" tells the user something was sent, and "stop" ends it. The
  key press on permission requests remains the safety boundary: a false task costs tokens, and
  edits if the harness is set to allow them without asking, but not commands the harness asks
  about.
- **Missed wake-ups.** If the transcriber writes the wake word in a way the list does not cover,
  the task is ignored. The log shows what was heard, and the list is one constant.
- **Cancel speed varies by harness.** The protocol says "as soon as possible". The 10 s kill is
  the backstop.
- **Permission behaviour is the harness's.** Which actions ask first depends on the harness's own
  settings, not on MacEcho.
- **Codex and Pi need adapters** that are not installed here. They are reached through
  `drive -- <command>` and are untested.
- **The protocol is young.** MacEcho uses six messages of version 1. If more is needed, the
  official Python SDK (`agent-client-protocol`) replaces `acp.py`.
