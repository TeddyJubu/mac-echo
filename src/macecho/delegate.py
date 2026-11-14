"""Hand a spoken request to Claude Code in the background and get back a summary to say aloud.

The voice model stays the talker; Claude Code (`claude -p`) does the work. Claude is asked to end
its reply with a SPOKEN: line, so the hard condensing happens in the smart model and the voice only
reads it out. Each Delegate keeps Claude's session id, so a follow-up ("now make it blue") continues
the same session.
"""
import asyncio
import json
import time
from dataclasses import dataclass

from macecho.daemon.voice import sentences, speakable

SPOKEN = "SPOKEN:"
INSTRUCTIONS = (
    "This request was spoken aloud and your reply will be read out by a voice assistant. "
    f"Finish with one last line that starts with {SPOKEN} and says what you did in one or two "
    "short, plain sentences: no code, file paths, lists or markdown."
)


@dataclass
class Result:
    ok: bool
    spoken: str            # what the voice says
    text: str = ""         # Claude's full reply, for the screen and the log
    seconds: float = 0.0   # wall time, process start to exit
    cost_usd: float = 0.0
    session_id: str | None = None


def spoken_summary(text: str) -> str:
    """The SPOKEN: line if Claude wrote one, else the first two sentences of its reply."""
    for line in reversed(text.splitlines()):
        if line.strip().startswith(SPOKEN):
            return speakable(line.strip()[len(SPOKEN):])
    # ponytail: two sentences of an unsummarised reply can still be long; cap words if that happens.
    return " ".join(sentences(speakable(text))[:2])


class Delegate:
    """Runs one Claude Code task at a time in `cwd`, continuing the same session each time."""

    def __init__(self, cwd: str, command=("claude",), permission_mode: str = "acceptEdits"):
        self.cwd, self.command, self.permission_mode = cwd, tuple(command), permission_mode
        self.session_id: str | None = None

    async def run(self, request: str) -> Result:
        args = [*self.command, "-p", request, "--output-format", "json",
                "--permission-mode", self.permission_mode, "--append-system-prompt", INSTRUCTIONS]
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
