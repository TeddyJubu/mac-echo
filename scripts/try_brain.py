"""Talk to the new brain from the terminal: Claude Haiku answers and hands coding work to Claude Code.

    ANTHROPIC_API_KEY=... .venv/bin/python scripts/try_brain.py [--project DIR] [--quiet]

Type what you would say. Replies are printed with the time to their first sentence, and spoken in
the macOS voice unless --quiet. A request for work starts Claude Code in the project folder (by
default a throwaway git repo, so nothing of yours changes); keep talking while it works, and Haiku
tells you how it went when it's done. Ctrl-D or "bye" to quit. Needs the network, so no sandbox.
"""
import argparse
import asyncio
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macecho.brain import Brain, Task  # noqa: E402
from macecho.daemon.voice import SystemVoice  # noqa: E402
from macecho.delegate import Delegate  # noqa: E402

GOODBYES = {"bye", "goodbye", "quit", "exit"}


def scratch_project() -> str:
    """A tiny git repo for Claude Code to work in, so trying this can't touch real code."""
    path = Path(tempfile.mkdtemp(prefix="macecho-try-"))
    (path / "app.py").write_text('def greet(name):\n    return "Hello " + name\n')
    git = ["git", "-c", "user.name=MacEcho", "-c", "user.email=macecho@localhost"]
    for command in (["init", "-q"], ["add", "."], ["commit", "-qm", "Start"]):
        subprocess.run(git + command, cwd=path, check=True)
    return str(path)


class Session:
    """What the Mac app will do, minus the audio: one brain turn at a time, Claude in the background."""

    def __init__(self, project: str, speak: bool):
        self.brain, self.project, self.speak = Brain(), project, speak
        self.lock, self.delegates, self.jobs = asyncio.Lock(), {}, set()

    async def say(self, make_turn):
        async with self.lock:  # a report waits for the reply being spoken, as in the app
            start = time.perf_counter()
            try:
                async for item in make_turn():
                    if isinstance(item, Task):
                        print(f"   🤖 handing to Claude Code [{item.session}]: {item.task}", flush=True)
                        self.start(item)
                        continue
                    if start is not None:
                        print(f"   ({time.perf_counter() - start:.2f}s to the first sentence)")
                        start = None
                    print(f"🗣  {item}", flush=True)
                    if self.speak:
                        await asyncio.to_thread(SystemVoice().speak, item)
            except Exception as e:  # no key, no network: show it and keep the session going
                print(f"   (brain error: {e})", flush=True)

    def start(self, task: Task):
        delegate = self.delegates.setdefault(task.session, Delegate(self.project))
        job = asyncio.create_task(self.run(delegate, task))
        self.jobs.add(job)
        job.add_done_callback(self.jobs.discard)

    async def run(self, delegate: Delegate, task: Task):
        result = await delegate.run(task.task)
        print(f"\n   🤖 Claude Code [{task.session}] {'finished' if result.ok else 'failed'} in "
              f"{result.seconds:.0f}s, ${result.cost_usd:.2f}. Its report:\n   {result.text}\n", flush=True)
        await self.say(lambda: self.brain.report(task, result.text if result.ok else result.spoken))


async def main(args):
    project = args.project or scratch_project()
    print(f"Claude Code works in: {project}\nType what you'd say. Ctrl-D or 'bye' to quit.\n")
    session = Session(project, speak=not args.quiet)
    while True:
        try:
            text = (await asyncio.to_thread(input, "you> ")).strip()
        except EOFError:
            break
        words = set(re.findall(r"[a-z]+", text.lower()))
        if words and words <= GOODBYES:  # "bye", "bye bye", "Goodbye!", "quit"
            break
        if text:
            await session.say(lambda: session.brain.reply(text))
    if session.jobs:
        print("Waiting for Claude Code to finish...", flush=True)
        await asyncio.gather(*list(session.jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", help="folder Claude Code works in (default: a throwaway git repo)")
    parser.add_argument("--quiet", action="store_true", help="print only; don't speak")
    asyncio.run(main(parser.parse_args()))
