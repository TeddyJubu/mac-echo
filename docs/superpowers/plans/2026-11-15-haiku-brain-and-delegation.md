# Haiku Brain and Bomby-Style Delegation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the Mac app, LFM3 keeps hearing and speaking, Claude Haiku 5.5 does the thinking, and Haiku hands coding work to Claude Code through one tool, `start_task`, then tells the user how it went.

**Architecture:** A new `brain.py` streams Haiku's reply sentence by sentence into LFM3's voice. When Haiku calls `start_task`, the app starts a `Delegate` (Claude Code, `claude -p`) in the background. Claude's report comes back to Haiku as a new turn, and Haiku says the outcome. `MACECHO_BRAIN=lfm3` (the default) keeps today's fully local behaviour.

**Tech Stack:** Python 3.12/3.13, `anthropic` 2.8 (`AsyncAnthropic`, `messages.stream`), mlx-vlm LFM3 (`SpeechModel.transcribe` / `SpeechModel.tts`), `claude -p` headless, pytest.

**Spec:** this plan carries its own spec, the decisions taken in conversation on 2026-11-15:
1. The Mac app's brain moves to Claude Haiku 5.5. LFM3 was not reliable enough: 78 % at best on `scripts/eval_router.py`, and it doesn't feel smart in conversation.
2. Three ideas from Bomby (github.com/bomby-app/bomby, MIT):
   - (a) The voice model has one tool, `start_task(task, session)`, and Claude's final report reaches it only as a later result.
   - (b) Claude Code gets Bomby's report rules: lead with the outcome, plain text, under ~100 words.
   - (c) Keep Claude Code sessions open (ACP or the Claude Agent SDK) instead of one `claude -p` process per task. This is adopted only if measuring shows a real gain.
3. LFM3 stays, as described under "Where LFM3 stands" below.

## How the three ideas fit MacEcho

| Bomby idea | Where it lands here | Task |
|---|---|---|
| (b) Report rules for the coding agent | `delegate.py` `INSTRUCTIONS` replaces our `SPOKEN:` line. The `route()` stub is deleted, because Haiku's tool replaces it. | 1 |
| (a) One tool, results only as tool results | `brain.py` `START_TASK`. Haiku sees Claude's report as its next turn, never as progress chatter. | 2, 4 |
| (a) Named sessions, follow-ups join their session | `Brain` names the session. The app keeps one `Delegate` per name, and a follow-up waits for its session (Task 1's lock). | 1, 4 |
| (c) Kept-open agent sessions | A measurement first (Task 5). Bomby's ACP client is ~1,500 lines plus an npm adapter. The Claude Agent SDK does the same for Claude alone, but its wheel bundles a 347 MB copy of Claude Code. Neither is worth it unless kept-open sessions save seconds per task. | 5 |

The Claude Code voice mode (`macecho listen/say`, the daemon, the skill) is untouched. There, Claude Code is already the brain and LFM3 is already only ears and a voice, which is the same split this plan gives the Mac app.

## Where LFM3 stands

- **It keeps two jobs:** hearing (`Perform ASR.`, 0.26–0.45 s for 2–4 s of speech) and speaking (`Perform TTS.`, 0.15–0.20 s to first audio). Both are measured on this Mac, and keeping them needs no new local model.
- **It loses the thinking whenever the Mac is online and `MACECHO_BRAIN=haiku`.**
- **It stays the offline, fully private brain** (`MACECHO_BRAIN=lfm3`, the default), the only mode where no text leaves the Mac.
- **Under review, in a separate plan:** FluidSpeech Pro for hearing and Zonos3-mini for speaking, adopted only if they beat LFM3 on this Mac. If both win, LFM3's last job is the offline brain. Whether that is worth ~2 GB of memory is the user's call then, not now.

## Global Constraints

- Python 3.12 or 3.13 only (onnxruntime has no newer wheels).
- Model id `claude-haiku-5-5`; `thinking={"type": "disabled"}` on every voice turn (allowed on Haiku 5.5 at effort high or below; the default is medium).
- `anthropic` goes in a new optional extra `brain`, never in the base install. The CLI stays standard-library only.
- `MACECHO_BRAIN` defaults to `lfm3`: nothing changes for anyone who doesn't opt in.
- Audio never leaves the Mac. With `MACECHO_BRAIN=haiku` the transcript text goes to Anthropic.
- The API key comes from `se secret request` into the environment or the repo's `.env`, never pasted in chat or committed.
- MLX tests, socket tests and anything using the network run with the Claude Code sandbox off.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Credit Bomby (MIT) where its wording is adapted.

## Review Focus

1. **The user talks over a Claude report:** the report's speech stops and the user gets an answer, but the report stays in Haiku's history, so "what did Claude say?" still works. Test: Task 4 `test_talking_over_a_report_cuts_it_short_without_losing_the_task`.
2. **No API key or no network:** the app says it can't reach Claude instead of going silent. Test: Task 4 `test_a_brain_failure_is_said_out_loud`.
3. **A follow-up to a session that is still working** waits, then continues the same Claude session rather than starting a second one in parallel. Test: Task 1 `test_a_follow_up_sent_while_busy_waits_then_continues_the_session`.
4. **Two tasks in one reply** (Haiku calls `start_task` twice): both start, and both calls are answered in a single message, which the API requires. Test: Task 2 `test_two_tasks_in_one_reply_both_start_and_both_are_answered`.
5. **A noise blip transcribed as empty text** makes no API call and says nothing. Test: Task 4 `test_an_empty_transcript_costs_nothing`.

---

### Task 1: Delegate: Bomby's report rules, and follow-ups wait for their session

**Files:**
- Modify: `src/macecho/delegate.py` (`INSTRUCTIONS`, `spoken_summary`, `Delegate.run`; delete `SPOKEN` and the uncommitted `route()` stub)
- Modify: `tests/test_delegate.py`
- Modify: `scripts/benchmark_delegate.py` (the `SPOKEN` import and the `spoken_line_missing` metric)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Delegate(cwd: str, command=("claude",), permission_mode="acceptEdits")` with `async run(request: str) -> Result`. `run` is now safe to call twice at once: the second call waits. `Result(ok, spoken, text, seconds, cost_usd, session_id)` is unchanged. `Result.text` is now a plain-text report of ~100 words, written to be spoken.

- [ ] **Step 1: Write the failing tests**

In `tests/test_delegate.py`, change the `ok` branch of `FAKE` to:

```python
if mode == "ok":
    print(json.dumps({"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 0.25,
                      "result": "I added a **dark mode** toggle to settings. Tests pass. I also tidied the CSS.\n\n```css\n.dark {}\n```"}))
```

Replace `test_success_speaks_the_spoken_line_and_keeps_the_full_reply` with:

```python
def test_success_speaks_the_outcome_and_keeps_the_full_report(fake):
    result = asyncio.run(fake("ok").run("add a dark mode toggle"))
    assert result.ok and result.cost_usd == 0.25 and result.session_id == "s-1"
    assert result.spoken == "I added a dark mode toggle to settings. Tests pass."  # markdown gone
    assert "```css" in result.text  # the screen still gets everything
    args = fake.calls()[0]
    assert args[:2] == ["-p", "add a dark mode toggle"] and "--resume" not in args
    assert args[args.index("--output-format") + 1] == "json"
    briefing = args[args.index("--append-system-prompt") + 1]
    assert "lead with the outcome" in briefing and "100 words" in briefing
    allowed = args[args.index("--allowedTools") + 1:]
    assert "Bash(pytest:*)" in allowed and "Bash(npm test:*)" in allowed
    assert all(rule.startswith("Bash(") for rule in allowed)  # test runners only, nothing broader
```

Add:

```python
def test_a_follow_up_sent_while_busy_waits_then_continues_the_session(fake):
    delegate = fake("ok")

    async def go():
        await asyncio.gather(delegate.run("first"), delegate.run("now make it blue"))

    asyncio.run(go())
    first, second = fake.calls()
    assert "--resume" not in first  # without waiting, both would start fresh sessions
    assert second[second.index("--resume") + 1] == "s-1"
```

Replace `test_summary_falls_back_to_two_sentences_without_a_spoken_line` with:

```python
def test_the_summary_is_the_first_two_sentences():
    assert spoken_summary("Done. Tests pass. Also refactored a lot.") == "Done. Tests pass."
    assert spoken_summary("**Fixed** the `load` bug.") == "Fixed the load bug."
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_delegate.py -v`
Expected: FAIL on `lead with the outcome` (still the SPOKEN briefing) and on `--resume not in first` (both runs start at once).

- [ ] **Step 3: Implement**

In `src/macecho/delegate.py`, replace the module docstring's second paragraph, `SPOKEN`, `INSTRUCTIONS` and `spoken_summary` with:

```python
"""Hand a spoken request to Claude Code in the background and get back a report to say aloud.

The voice stays the talker; Claude Code (`claude -p`) does the work. Claude is briefed to write its
final message for the ear: outcome first, plain text, short. Each Delegate keeps Claude's session
id, so a follow-up ("now make it blue") continues the same session, and one sent while Claude is
still working waits for it.
"""
```

```python
# Adapted from Bomby's harness briefing (MIT, github.com/bomby-app/bomby, agent/prompts/harness-briefing.md).
INSTRUCTIONS = (
    "You are working for a voice assistant. The user talks to the assistant; it hands you tasks, and "
    "your final message goes back to it and is read aloud. Nobody can answer questions while you "
    "work, so make reasonable decisions yourself and say which ones you made. Your final message "
    "must lead with the outcome in one or two sentences, be plain text (no markdown, tables, code "
    "blocks or file trees), name files, commands or numbers only when the user needs them, and stay "
    "under about 100 words, unless the task is to find information the user asked for."
)
```

```python
def spoken_summary(text: str) -> str:
    """The first two sentences: the briefing asks Claude to lead with the outcome."""
    # ponytail: two sentences of a report that ignored the briefing can still be long; cap words then.
    return " ".join(sentences(speakable(text))[:2])
```

In `Delegate.__init__` add:

```python
        self._turn = asyncio.Lock()  # one task at a time per session: a follow-up waits its turn
```

Rename the current `async def run(self, request: str) -> Result:` to `async def _run(self, request: str) -> Result:` and add above it:

```python
    async def run(self, request: str) -> Result:
        async with self._turn:
            return await self._run(request)
```

Delete the whole `route()` function at the end of the file.

In `scripts/benchmark_delegate.py`:
- Change the import to `from macecho.delegate import Delegate  # noqa: E402`.
- In `BUDGET`, replace `"spoken_line_missing": 0,     # Claude ignored the SPOKEN: instruction` with `"long_reports": 0,            # over 100 words: Claude ignored the briefing`.
- Replace the row field `"spoken_line": any(l.strip().startswith(SPOKEN) for l in result.text.splitlines()),` with `"report_words": len(result.text.split()),`.
- Replace `metrics["spoken_line_missing"] = sum(not t["spoken_line"] for t in tasks)` with `metrics["long_reports"] = sum(t["report_words"] > 100 for t in tasks)`.

- [ ] **Step 4: Run the tests to see them pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_delegate.py -v && .venv/bin/python -c "import ast,sys; ast.parse(open('scripts/benchmark_delegate.py').read())"`
Expected: all pass; the benchmark still parses. Also run `grep -n SPOKEN -r src scripts tests`, which should print nothing.

- [ ] **Step 5: Commit**

```bash
git add src/macecho/delegate.py tests/test_delegate.py scripts/benchmark_delegate.py
git commit -m "feat: Claude reports for the ear, and follow-ups wait for their session

Briefing adapted from Bomby (MIT). route() is gone: the brain's start_task tool decides.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The brain: Claude Haiku with one tool

**Files:**
- Create: `src/macecho/brain.py`
- Create: `tests/test_brain.py`
- Modify: `pyproject.toml` (add the `brain` extra)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `MODEL = "claude-haiku-5-5"`
  - `@dataclass(frozen=True) class Task: id: str; task: str; session: str`
  - `Brain(client=None, model: str = MODEL, can_delegate: bool = True)` with these methods:
    - `async reply(user_text: str)` is an async generator yielding `str` sentences and `Task` items.
    - `async report(task: Task, text: str)` has the same yield types.
  - Within one turn, every `Task` comes before the last sentence. This lets the app start Claude before it speaks "On it.", so a barge-in during those words can't lose the task.

- [ ] **Step 1: Add the extra and install it**

In `pyproject.toml` under `[project.optional-dependencies]`, add:

```toml
# MACECHO_BRAIN=haiku (macecho.brain): Claude Haiku thinks; LFM3 still hears and speaks.
brain = ["anthropic>=2.8"]
```

Run: `UV_CACHE_DIR="$TMPDIR/uv-cache" uv pip install --python .venv/bin/python -e ".[app,brain]" pytest`
Expected: installs `anthropic`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_brain.py`:

```python
import asyncio
from types import SimpleNamespace as NS

from macecho.brain import MODEL, Brain, Task


class FakeStream:
    """What `client.messages.stream(...)` returns: text deltas, then the whole message."""

    def __init__(self, deltas, content):
        self.deltas, self.content = deltas, content

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for delta in self.deltas:
                yield delta
        return gen()

    async def get_final_message(self):
        return NS(content=self.content)


class FakeClient:
    """Plays back one (deltas, content) pair per turn, and records what each turn was sent."""

    def __init__(self, *turns):
        self.turns, self.calls, self.messages = list(turns), [], self

    def stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # a copy: history grows
        return FakeStream(*self.turns.pop(0))


def text(t):
    return NS(type="text", text=t)


def tool(task, session, id="tu_1"):
    return NS(type="tool_use", id=id, name="start_task", input={"task": task, "session": session})


def collect(agen):
    async def go():
        return [item async for item in agen]
    return asyncio.run(go())


def test_sentences_come_out_as_soon_as_they_end():
    client = FakeClient((["Tokyo is the cap", "ital. It has ", "14 million people."],
                         [text("Tokyo is the capital. It has 14 million people.")]))
    assert collect(Brain(client).reply("Capital of Japan?")) == [
        "Tokyo is the capital.", "It has 14 million people."]
    sent = client.calls[0]
    assert sent["model"] == MODEL and sent["thinking"] == {"type": "disabled"}
    assert sent["messages"] == [{"role": "user", "content": "Capital of Japan?"}]
    assert [t["name"] for t in sent["tools"]] == ["start_task"]


def test_a_task_starts_before_the_last_words_are_spoken():
    client = FakeClient((["On it."], [text("On it."), tool("Add a dark mode toggle.", "dark-mode")]))
    brain = Brain(client)
    assert collect(brain.reply("Add dark mode")) == [
        Task("tu_1", "Add a dark mode toggle.", "dark-mode"), "On it."]
    answer = brain.history[-1]
    assert answer["role"] == "user" and answer["content"][0]["tool_use_id"] == "tu_1"


def test_two_tasks_in_one_reply_both_start_and_both_are_answered():
    client = FakeClient(([], [tool("Fix the build.", "build", "tu_1"), tool("Bump the version.", "version", "tu_2")]))
    brain = Brain(client)
    assert [t.session for t in collect(brain.reply("Fix the build and bump the version"))] == ["build", "version"]
    assert [r["tool_use_id"] for r in brain.history[-1]["content"]] == ["tu_1", "tu_2"]  # one message


def test_a_report_is_a_new_turn_that_haiku_answers():
    client = FakeClient((["On it."], [text("On it."), tool("Add dark mode.", "dark-mode")]),
                        (["Dark mode is in."], [text("Dark mode is in.")]))
    brain = Brain(client)
    task = [item for item in collect(brain.reply("Add dark mode")) if isinstance(item, Task)][0]
    assert collect(brain.report(task, "Added the toggle to settings.")) == ["Dark mode is in."]
    assert "Added the toggle to settings." in str(client.calls[1]["messages"][-1])


def test_without_a_project_folder_there_is_no_tool():
    client = FakeClient((["Hi."], [text("Hi.")]))
    collect(Brain(client, can_delegate=False).reply("Hi"))
    assert client.calls[0]["tools"] == [] and "no project folder" in client.calls[0]["system"]
```

- [ ] **Step 3: Run the tests to see them fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_brain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'macecho.brain'`.

- [ ] **Step 4: Implement**

Create `src/macecho/brain.py`:

```python
"""The Mac app's brain: Claude Haiku in the cloud, between LFM3's ears and LFM3's voice.

LFM3 still hears ("Perform ASR.") and speaks ("Perform TTS."); only the thinking moves here, and
with it the choice of when to hand work to Claude Code, through Haiku's one tool, start_task
(the shape Bomby uses: github.com/bomby-app/bomby, MIT). The user's words go to Anthropic as
text; their audio never leaves the Mac.
"""
import re
from dataclasses import dataclass

MODEL = "claude-haiku-5-5"
SYSTEM = ("You are MacEcho, a voice assistant on the user's Mac. Everything you write is read "
          "aloud, so answer in one to three short, plain sentences: no markdown, lists or code. ")
DELEGATING = (
    "Claude Code, a coding agent, works in the user's project folder. When the user asks for "
    "anything that needs their files, code, shell or git, including questions only the project "
    "can answer, say in a few words that you're handing it over, then call start_task. Don't call "
    "it for small talk or for questions you can answer yourself. Claude's report arrives later as "
    "a message; tell the user the outcome in a sentence or two.")
ALONE = ("There is no project folder set up, so you can't change or check the user's files; "
         "say so briefly if they ask.")
START_TASK = {
    "name": "start_task",
    "description": ("Hand a task to Claude Code, which works in the user's project with its files, "
                    "shell and git. It runs in the background and can take minutes; its report "
                    "arrives later as a message."),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": (
                "What the user wants, with every detail they gave, as one clear request. Claude "
                "Code can't hear this conversation.")},
            "session": {"type": "string", "description": (
                "A short name for the piece of work, like \"dark-mode\". Reuse it for follow-ups "
                "to that work; pick a new one for unrelated work.")},
        },
        "required": ["task", "session"],
    },
}
STARTED = "Started. Claude Code is working on it; its report will arrive as a later message."
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Task:
    id: str        # the tool call's id
    task: str
    session: str


class Brain:
    """One call's conversation with Haiku. Not safe for two turns at once: the caller serialises."""

    def __init__(self, client=None, model: str = MODEL, can_delegate: bool = True):
        if client is None:
            import anthropic  # the brain extra; reads ANTHROPIC_API_KEY
            client = anthropic.AsyncAnthropic()
        self.client, self.model = client, model
        self.tools = [START_TASK] if can_delegate else []
        self.system = SYSTEM + (DELEGATING if can_delegate else ALONE)
        # ponytail: the whole call's history goes with every turn; trim the oldest turns if calls get long.
        self.history: list[dict] = []

    async def reply(self, user_text: str):
        """Sentences to speak as they arrive, and a Task for each start_task call."""
        self.history.append({"role": "user", "content": user_text})
        async for item in self._turn():
            yield item

    async def report(self, task: Task, text: str):
        """Claude Code finished: a new turn in which Haiku tells the user how it went."""
        self.history.append({"role": "user", "content": f"[Claude Code's report, session {task.session}]\n{text}"})
        async for item in self._turn():
            yield item

    async def _turn(self):
        pending = ""
        async with self.client.messages.stream(
                model=self.model, max_tokens=400, system=self.system, tools=self.tools,
                thinking={"type": "disabled"}, messages=self.history) as stream:
            async for delta in stream.text_stream:
                *done, pending = SENTENCE_END.split(pending + delta)  # the last piece may not have ended
                for sentence in done:
                    if sentence.strip():
                        yield sentence.strip()
            message = await stream.get_final_message()

        content = [{"type": "text", "text": b.text} if b.type == "text" else
                   {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                   for b in message.content if (b.type == "text" and b.text) or b.type == "tool_use"]
        if content:
            self.history.append({"role": "assistant", "content": content})
        tasks = [Task(b.id, b.input["task"], b.input["session"]) for b in message.content if b.type == "tool_use"]
        if tasks:  # the API needs every tool call answered in the very next message
            self.history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": t.id, "content": STARTED} for t in tasks]})
        # Tasks before the last words: the app starts Claude first, so talking over "On it." can't lose it.
        for task in tasks:
            yield task
        if pending.strip():
            yield pending.strip()
```

- [ ] **Step 5: Run the tests to see them pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_brain.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/macecho/brain.py tests/test_brain.py
git commit -m "feat: a Claude Haiku brain with one tool, start_task

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Brain benchmark: the go/no-go before wiring it in

**Files:**
- Create: `scripts/benchmark_brain.py`
- Create: `benchmarks/brain-haiku.json` (its first run)

**Interfaces:**
- Consumes: `Brain`, `Task`, `MODEL` from Task 2; `CASES` (40 `(text, "CLAUDE"|"CHAT")` pairs) from `scripts/eval_router.py`.
- Produces: a baseline file and a decision. **If accuracy < 0.95 or the p50 time to first sound is over 0.8 s, stop after this task and report to the user.** Task 4 waits for their call: try Muse Ember via `anthropic.AsyncAnthropic(base_url=..., api_key=...)` (Meta lists a Messages endpoint; compatibility unverified), or adjust the prompt.

- [ ] **Step 1: Get the key**

Run `se secret request` (tool timeout ≥ 10 minutes) for `ANTHROPIC_API_KEY`, then export it into the benchmark command's environment. Never print it, write it to a committed file or put it in chat.

- [ ] **Step 2: Write the benchmark**

Create `scripts/benchmark_brain.py`:

```python
"""Brain benchmark: is Claude Haiku quick enough to talk, and right about what goes to Claude Code?

    .venv/bin/python scripts/benchmark_brain.py [--model claude-haiku-5-5] [--out FILE]

The 40 labelled requests from eval_router.py go through the real Brain, each as the first turn of a
fresh call, one at a time on one shared connection (as in the app). LFM3 scored 78 % at best on the
same requests (benchmarks/router-lfm3-*.json). Needs ANTHROPIC_API_KEY and the network, so no
sandbox; costs well under a cent.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from eval_router import CASES  # noqa: E402
from macecho.brain import MODEL, Brain, Task  # noqa: E402

# Metric -> limit. accuracy is a floor; the rest are ceilings. Anything out flags OUT and exits 1.
BUDGET = {
    "accuracy": 0.95,           # floor: LFM3's best was 0.78
    "first_sound_p50_s": 0.8,   # request sent -> first sentence (or hand-over) ready to speak
    "first_sound_p95_s": 1.5,
}


async def one(client, model, text):
    start, first, got = time.perf_counter(), None, "CHAT"
    async for item in Brain(client, model).reply(text):
        if first is None:
            first = time.perf_counter() - start
        if isinstance(item, Task):
            got = "CLAUDE"
    return got, first if first is not None else time.perf_counter() - start


async def run(model):
    import anthropic
    client, rows = anthropic.AsyncAnthropic(), []
    for text, expected in CASES:
        got, first = await one(client, model, text)
        rows.append({"text": text, "expected": expected, "got": got, "first_s": round(first, 3)})
        print(f"  {'ok ' if got == expected else 'BAD'} {first:5.2f}s  {expected:6} -> {got:6}  {text}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()

    rows = asyncio.run(run(args.model))
    quantiles = statistics.quantiles([r["first_s"] for r in rows], n=20)  # [9] median, [18] p95
    metrics = {
        "accuracy": round(sum(r["got"] == r["expected"] for r in rows) / len(rows), 3),
        "chat_sent_to_claude": sum(r["expected"] == "CHAT" and r["got"] == "CLAUDE" for r in rows),
        "claude_kept_as_chat": sum(r["expected"] == "CLAUDE" and r["got"] == "CHAT" for r in rows),
        "first_sound_p50_s": round(quantiles[9], 3),
        "first_sound_p95_s": round(quantiles[18], 3),
    }
    print(f"\n{args.model} on {len(rows)} requests:\n\nMetric                     value   limit")
    out = []
    for name, value in metrics.items():
        limit = BUDGET.get(name)
        bad = limit is not None and (value < limit if name == "accuracy" else value > limit)
        out += [name] if bad else []
        print(f"  {name:22} {value!s:>8}  {'' if limit is None else limit!s:>6}  {'OUT' if bad else ''}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "metrics": metrics, "budget": BUDGET,
                                              "rows": rows}, indent=2))
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run (sandbox off, key in the environment): `.venv/bin/python scripts/benchmark_brain.py --out benchmarks/brain-haiku.json`
Expected: 40 lines, then the metrics table. Exit 0 means Task 4 goes ahead. Exit 1 means stop and report the misses and numbers to the user.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_brain.py benchmarks/brain-haiku.json
git commit -m "bench: is Haiku quick and right enough to be the voice's brain?

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Wire the brain into the Mac app

**Files:**
- Modify: `src/macecho/config.py` (`MacEchoConfig`: `brain`, `brain_model`, `project_dir`)
- Modify: `src/macecho/agent.py` (imports, `Agent.__init__`, `_process_speech_segment`; new `_process_with_brain`, `_speak_all`, `_speak`, `_start_task`, `_run_task`, `_deliver_report`)
- Modify: `CLAUDE.md` (how the brain fits)
- Create: `tests/test_agent_brain.py`

**Interfaces:**
- Consumes: `Brain`, `Task` (Task 2); `Delegate`, `Result` (Task 1).
- Produces: env settings `MACECHO_BRAIN=lfm3|haiku`, `MACECHO_BRAIN_MODEL` and `MACECHO_PROJECT_DIR`, plus the module constant `agent.CANT_THINK`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agent_brain.py`:

```python
import asyncio

import numpy as np

from macecho.agent import CANT_THINK, Agent
from macecho.brain import Task
from macecho.delegate import Result


class FakeSpeech:
    def __init__(self, heard="Add dark mode"):
        self.heard, self.said = heard, []

    def transcribe(self, audio):
        return self.heard

    def tts(self, text, stop=None):
        self.said.append(text)
        yield np.zeros(240, np.float32)


class FakeBrain:
    def __init__(self, fail=False, long_report=False):
        self.fail, self.long_report, self.asked, self.reports = fail, long_report, [], []

    async def reply(self, text):
        self.asked.append(text)
        if self.fail:
            raise RuntimeError("no API key")
        yield Task("tu_1", "Add a dark mode toggle.", "dark-mode")
        yield "On it."

    async def report(self, task, text):
        self.reports.append((task.session, text))
        yield "Dark mode is in."
        if self.long_report:
            await asyncio.sleep(10)
            yield "And more."


class FakeDelegate:
    async def run(self, request):
        return Result(True, "Added it.", "Added the dark mode toggle to settings.")


def make_agent(**brain):
    agent = object.__new__(Agent)  # no audio devices, VAD or model: just the brain path
    agent.speech, agent.brain = FakeSpeech(), FakeBrain(**brain)
    agent.delegates, agent.background = {"dark-mode": FakeDelegate()}, set()
    agent.brain_lock, agent.audio_player_queue = asyncio.Lock(), asyncio.Queue()
    agent.current_llm_task = None
    return agent


def test_a_task_is_acknowledged_now_and_reported_when_claude_finishes():
    async def go():
        agent = make_agent()
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        await asyncio.gather(*agent.background)
        return agent

    agent = asyncio.run(go())
    assert agent.speech.said == ["On it.", "Dark mode is in."]
    assert agent.brain.reports == [("dark-mode", "Added the dark mode toggle to settings.")]
    assert agent.audio_player_queue.qsize() == 2  # one chunk of speech each


def test_talking_over_a_report_cuts_it_short_without_losing_the_task():
    async def go():
        agent = make_agent(long_report=True)
        job = asyncio.create_task(agent._deliver_report(Task("tu_1", "x", "dark-mode"), "Added it."))
        while not agent.speech.said:
            await asyncio.sleep(0.01)
        agent.current_llm_task.cancel()  # what _cancel_ongoing_processing does when the user talks
        await asyncio.wait_for(job, 2)   # the delivery ends quietly instead of raising
        return agent

    agent = asyncio.run(go())
    assert agent.speech.said == ["Dark mode is in."]
    assert agent.brain.reports == [("dark-mode", "Added it.")]  # Haiku has it, for "what did Claude say?"


def test_a_brain_failure_is_said_out_loud():
    async def go():
        agent = make_agent(fail=True)
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent

    assert asyncio.run(go()).speech.said == [CANT_THINK]


def test_an_empty_transcript_costs_nothing():
    async def go():
        agent = make_agent()
        agent.speech.heard = ""
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent

    agent = asyncio.run(go())
    assert agent.brain.asked == [] and agent.speech.said == []
```

- [ ] **Step 2: Run the tests to see them fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_brain.py -v`
Expected: FAIL with `ImportError: cannot import name 'CANT_THINK'`.

- [ ] **Step 3: Add the settings**

In `src/macecho/config.py`, add `Literal` to the `typing` import, and add this after the `speech_model` field of `MacEchoConfig`:

```python
    brain: Literal["lfm3", "haiku"] = Field(
        default="lfm3",
        description="Who thinks in the Mac app. lfm3: the speech model, local and offline. haiku: "
                    "Claude Haiku via Anthropic's API (needs ANTHROPIC_API_KEY and the brain extra); "
                    "LFM3 still hears and speaks, and the transcript text goes to Anthropic.")
    brain_model: str = Field(default="claude-haiku-5-5", description="Claude model when brain is haiku.")
    project_dir: Optional[str] = Field(
        default=None,
        description="Folder Claude Code works in when the haiku brain hands it a task. Unset: no hand-overs.")
```

- [ ] **Step 4: Implement the brain path in `agent.py`**

Add to the imports:

```python
import contextlib
from macecho.brain import Task
from macecho.delegate import Delegate
```

Add a module constant after the imports:

```python
CANT_THINK = "I can't reach Claude right now."
```

In `Agent.__init__`, directly after the `if speech is not None: ... else: self._init_legacy_pipeline(config)` block, add:

```python
        self.brain = None  # Claude Haiku when MACECHO_BRAIN=haiku; LFM3 then only hears and speaks
        self.delegates = {}  # Claude Code sessions by name, for the brain's start_task
        self.background = set()  # running Claude Code tasks (held, or asyncio may drop them)
        self.brain_lock = asyncio.Lock()  # one brain turn at a time: replies and reports share history
        if config.brain == "haiku":
            if self.speech is None:
                raise ValueError("MACECHO_BRAIN=haiku needs MACECHO_SPEECH_MODEL: LFM3 hears and speaks for it")
            from macecho.brain import Brain
            self.brain = Brain(model=config.brain_model, can_delegate=bool(config.project_dir))
```

In `_process_speech_segment`, replace

```python
                self.current_llm_task = asyncio.create_task(
                    self._process_speech_model(speech_segment, correlation_id))
```

with

```python
                handle = self._process_with_brain if self.brain else self._process_speech_model
                self.current_llm_task = asyncio.create_task(handle(speech_segment, correlation_id))
```

Add these methods after `_process_speech_model`:

```python
    async def _process_with_brain(self, speech_segment: np.ndarray, correlation_id: str):
        """LFM3 hears, Claude Haiku thinks (and may hand work to Claude Code), LFM3 speaks."""
        audio = speech_segment.astype(np.float32)
        if speech_segment.dtype == np.int16:
            audio /= 32768.0
        start = time.time()
        text = await asyncio.to_thread(self.speech.transcribe, audio)
        print(f"👂 {text!r} after {time.time() - start:.2f}s")
        if not text:
            return
        async with self.brain_lock:
            try:
                await self._speak_all(self.brain.reply(text), start)
            except Exception as e:  # no key, no network, an API error: say so instead of going quiet
                print(f"Brain error: {e}")
                await self._speak(CANT_THINK)

    async def _speak_all(self, items, since: float | None = None):
        """One brain turn aloud: sentences in order; a hand-over starts Claude Code and moves on."""
        async for item in items:
            if isinstance(item, Task):
                self._start_task(item)
            else:
                await self._speak(item, since)
                since = None

    async def _speak(self, sentence: str, since: float | None = None):
        """One sentence in LFM3's voice, onto the player as it's generated. Cancelling stops it."""
        loop, stop = asyncio.get_running_loop(), threading.Event()

        def produce():
            nonlocal since
            for chunk in self.speech.tts(sentence, stop):
                if stop.is_set():
                    return
                if since is not None:
                    print(f"🔊 first audio after {time.time() - since:.2f}s")
                    since = None
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(self.audio_player_queue.put_nowait, pcm)

        try:
            await asyncio.to_thread(produce)
        except asyncio.CancelledError:
            stop.set()  # stops MLX generation within one step
            raise

    def _start_task(self, task: Task):
        """Claude Code starts on it in the background; barge-ins never cancel this."""
        if task.session not in self.delegates:
            self.delegates[task.session] = Delegate(self.config.project_dir)
        job = asyncio.create_task(self._run_task(self.delegates[task.session], task))
        self.background.add(job)
        job.add_done_callback(self.background.discard)

    async def _run_task(self, delegate, task: Task):
        print(f"🤖 {task.session}: {task.task}")
        result = await delegate.run(task.task)
        print(f"🤖 {task.session}: {'done' if result.ok else 'failed'} in {result.seconds:.0f}s, ${result.cost_usd:.2f}")
        await self._deliver_report(task, result.text if result.ok else result.spoken)

    async def _deliver_report(self, task: Task, text: str):
        """Haiku tells the user how Claude's task went, once any reply being spoken has finished."""
        # ponytail: also waits out a long reply; add a priority if reports start feeling late.
        async with self.brain_lock:
            speaking = asyncio.create_task(self._speak_all(self.brain.report(task, text)))
            self.current_llm_task = speaking  # the user can talk over it, as over any reply
            with contextlib.suppress(asyncio.CancelledError):
                await speaking
```

> **Learn-by-Doing point at execution:** `_deliver_report`'s timing policy (wait for the current reply, interrupt it, or wait for N seconds of silence) is a user-experience choice. Offer it to the user as a `TODO(human)` before writing the default above.

- [ ] **Step 5: Run the tests to see them pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_agent_brain.py tests/test_brain.py tests/test_delegate.py -v`
Expected: all pass. If importing `macecho.agent` fails inside the sandbox (PortAudio or onnxruntime), rerun with the sandbox off.

- [ ] **Step 6: Document it in `CLAUDE.md`**

Under "Core modules", after the `agent.py` bullet, add:

```markdown
- `brain.py`: with `MACECHO_BRAIN=haiku`, Claude Haiku thinks for the app (LFM3 still hears and
  speaks) and hands work to Claude Code (`delegate.py`) in `MACECHO_PROJECT_DIR` through its one
  tool, `start_task`. Needs `ANTHROPIC_API_KEY` and the `brain` extra. Default `lfm3` stays offline.
```

- [ ] **Step 7: Run the whole suite**

Run (sandbox off): `PYTHONPATH=src .venv/bin/python -m pytest tests/ -q`
Expected: everything passes.

- [ ] **Step 8: Try it for real**

Run (sandbox off; key exported; `MACECHO_SPEECH_MODEL` already set in `.env`): `MACECHO_BRAIN=haiku MACECHO_PROJECT_DIR="$PWD" PYTHONPATH=src .venv/bin/python -m macecho.menubar`
Ask the user to say "What's the capital of Japan?", then "Which branch am I on?", then talk over the report. Watch for the `👂`, `🔊 first audio after` and `🤖` lines in `~/Library/Logs/MacEcho.log` or the terminal.

- [ ] **Step 9: Commit**

```bash
git add src/macecho/config.py src/macecho/agent.py tests/test_agent_brain.py CLAUDE.md
git commit -m "feat: MACECHO_BRAIN=haiku: LFM3 hears and speaks, Haiku thinks and hands work to Claude

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Measure kept-open Claude sessions (the gate for idea c)

**Files:**
- Create: `scripts/probe_claude_session.py`
- Create: `benchmarks/claude-session-probe.json` (its run)

**Interfaces:**
- Consumes: the user's `claude` CLI and its login; `claude-agent-sdk` only in a throwaway environment (`uv run --with`), never as a project dependency.
- Produces: a decision. **If a kept-open session's median task time is at least 2 s below `claude -p`'s**, write a new plan to move `Delegate` onto `ClaudeSDKClient` (or ACP, if more agents than Claude are wanted). That would bring steering mid-task and `interrupt()`. Otherwise keep `claude -p` and record why in the commit message.

- [ ] **Step 1: Write the probe**

Create `scripts/probe_claude_session.py`:

```python
"""Is a kept-open Claude Code session quicker per task than one `claude -p` per task?

    UV_CACHE_DIR="$TMPDIR/uv-cache" uv run --with claude-agent-sdk python scripts/probe_claude_session.py [--runs 3] [--out FILE]

Decides whether Delegate moves to the Claude Agent SDK (or ACP, as Bomby does). The same trivial
prompt both ways, in a throwaway folder; uses the user's own `claude` and login. Needs the network,
so no sandbox; costs a few cents.
"""
import argparse
import asyncio
import json
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

PROMPT = "Reply with the single word OK."


def cold(cwd: str) -> float:
    start = time.monotonic()
    subprocess.run(["claude", "-p", PROMPT, "--output-format", "json"], cwd=cwd, capture_output=True, check=True)
    return time.monotonic() - start


async def warm(cwd: str, runs: int) -> tuple[float, list[float]]:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage

    start, times = time.monotonic(), []
    async with ClaudeSDKClient(ClaudeAgentOptions(cwd=cwd, cli_path=shutil.which("claude"))) as client:
        connect = time.monotonic() - start
        for _ in range(runs):
            start = time.monotonic()
            await client.query(PROMPT)
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    break
            times.append(time.monotonic() - start)
    return connect, times


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out")
    args = parser.parse_args()
    cwd = tempfile.mkdtemp(prefix="claude-probe-")
    cold_s = [cold(cwd) for _ in range(args.runs)]
    connect_s, warm_s = asyncio.run(warm(cwd, args.runs))
    result = {"cold_median_s": round(statistics.median(cold_s), 2),
              "warm_connect_s": round(connect_s, 2),
              "warm_median_s": round(statistics.median(warm_s), 2),
              "cold_s": [round(s, 2) for s in cold_s], "warm_s": [round(s, 2) for s in warm_s]}
    result["saved_per_task_s"] = round(result["cold_median_s"] - result["warm_median_s"], 2)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run (sandbox off): `UV_CACHE_DIR="$TMPDIR/uv-cache" uv run --with claude-agent-sdk python scripts/probe_claude_session.py --out benchmarks/claude-session-probe.json`
Expected: `saved_per_task_s` printed. It downloads the SDK once (~350 MB, because it bundles Claude Code) into the uv cache, not into the project.

- [ ] **Step 3: Commit with the decision**

```bash
git add scripts/probe_claude_session.py benchmarks/claude-session-probe.json
git commit -m "bench: kept-open Claude sessions save <N> s per task: <adopt in a new plan | keep claude -p>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Replace `<N>` and the verdict with the measured number and the decision before committing.
