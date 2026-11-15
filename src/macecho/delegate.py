"""Hand a spoken request to Claude Code in the background and get back a report to say aloud.

The voice stays the talker; Claude Code (`claude -p`) does the work. Claude is briefed to write its
final message for the ear: outcome first, plain text, short. Each Delegate keeps Claude's session
id, so a follow-up ("now make it blue") continues the same session, and one sent while Claude is
still working waits for it.
"""
import asyncio
import json
import time
from dataclasses import dataclass

from macecho.daemon.voice import sentences, speakable

# Adapted from Bomby's harness briefing (MIT, github.com/bomby-app/bomby, agent/prompts/harness-briefing.md).
INSTRUCTIONS = (
    "You are working for a voice assistant. The user talks to the assistant; it hands you tasks, and "
    "your final message goes back to it and is read aloud. Nobody can answer questions while you "
    "work, so make reasonable decisions yourself and say which ones you made. Your final message "
    "must lead with the outcome in one or two sentences, be plain text (no markdown, tables, code "
    "blocks or file trees), name files, commands or numbers only when the user needs them, and stay "
    "under about 100 words, unless the task is to find information the user asked for."
)
# Commands Claude may run without asking: test runners only, so it can check its own work.
# Everything else stays refused in a headless run (acceptEdits allows edits, not commands).
TEST_COMMANDS = ("pytest", "python -m pytest", "python3 -m pytest", "uv run pytest", "npm test",
                 "npm run test", "pnpm test", "yarn test", "go test", "cargo test", "swift test")


@dataclass
class Result:
    ok: bool
    spoken: str            # what the voice says
    text: str = ""         # Claude's full reply, for the screen and the log
    seconds: float = 0.0   # wall time, process start to exit
    cost_usd: float = 0.0
    session_id: str | None = None


def spoken_summary(text: str) -> str:
    """The first two sentences: the briefing asks Claude to lead with the outcome."""
    # ponytail: two sentences of a report that ignored the briefing can still be long; cap words then.
    return " ".join(sentences(speakable(text))[:2])


class Delegate:
    """Runs one Claude Code task at a time in `cwd`, continuing the same session each time."""

    def __init__(self, cwd: str, command=("claude",), permission_mode: str = "acceptEdits"):
        self.cwd, self.command, self.permission_mode = cwd, tuple(command), permission_mode
        self.session_id: str | None = None
        self._turn = asyncio.Lock()  # one task at a time per session: a follow-up waits its turn

    async def run(self, request: str) -> Result:
        async with self._turn:
            return await self._run(request)

    async def _run(self, request: str) -> Result:
        args = [*self.command, "-p", request, "--output-format", "json",
                "--permission-mode", self.permission_mode, "--append-system-prompt", INSTRUCTIONS,
                "--allowedTools", *(f"Bash({command}:*)" for command in TEST_COMMANDS)]
        if self.session_id:
            args += ["--resume", self.session_id]
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, cwd=self.cwd, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return Result(False, "I couldn't find Claude Code on this Mac.")
        try:
            out, err = await proc.communicate()
        except asyncio.CancelledError:  # the user called the task off: don't leave Claude running
            proc.kill()
            await proc.wait()
            raise
        seconds = time.monotonic() - start
        try:
            data = json.loads(out)
        except ValueError:
            reason = (err or out).decode(errors="replace").strip().splitlines()[-1:] or ["no output"]
            return Result(False, "Claude Code stopped with an error.", reason[0], seconds)

        self.session_id = data.get("session_id") or self.session_id
        text = data.get("result") or ""
        ok = proc.returncode == 0 and not data.get("is_error")
        spoken = spoken_summary(text) if ok else "Claude Code ran into a problem. The details are on screen."
        return Result(ok, spoken, text, seconds, data.get("total_cost_usd") or 0.0, self.session_id)

