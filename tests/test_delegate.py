import asyncio
import json
import sys
import time

import pytest

from macecho.delegate import Delegate, spoken_summary

# A stand-in for `claude -p`: logs its arguments, then behaves as the FAKE_MODE file says.
FAKE = r'''
import json, sys, time, pathlib
here = pathlib.Path(sys.argv[0]).parent
(here / "calls.jsonl").open("a").write(json.dumps(sys.argv[1:]) + "\n")
mode = (here / "mode").read_text()
if mode == "ok":
    print(json.dumps({"type": "result", "is_error": False, "session_id": "s-1", "total_cost_usd": 0.25,
                      "result": "Added the toggle.\n\n```css\n.dark {}\n```\nSPOKEN: I added a **dark mode** toggle to settings."}))
elif mode == "error":
    print(json.dumps({"type": "result", "is_error": True, "session_id": "s-1", "result": "API error"}))
elif mode == "crash":
    print("Traceback: boom", file=sys.stderr); sys.exit(1)
elif mode == "slow":
    time.sleep(30)
'''


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / "claude.py"
    script.write_text(FAKE)

    def make(mode):
        (tmp_path / "mode").write_text(mode)
        return Delegate(str(tmp_path), command=(sys.executable, str(script)))

    make.calls = lambda: [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    return make


def test_success_speaks_the_spoken_line_and_keeps_the_full_reply(fake):
    result = asyncio.run(fake("ok").run("add a dark mode toggle"))
    assert result.ok and result.cost_usd == 0.25 and result.session_id == "s-1"
    assert result.spoken == "I added a dark mode toggle to settings."  # markdown gone
    assert "```css" in result.text  # the screen still gets everything
    args = fake.calls()[0]
    assert args[:2] == ["-p", "add a dark mode toggle"] and "--resume" not in args
    assert args[args.index("--output-format") + 1] == "json"
    assert "SPOKEN:" in args[args.index("--append-system-prompt") + 1]


def test_a_follow_up_continues_the_same_claude_session(fake):
    delegate = fake("ok")
    asyncio.run(delegate.run("first"))
    asyncio.run(delegate.run("now make it blue"))
    second = fake.calls()[1]
    assert second[second.index("--resume") + 1] == "s-1"


def test_failures_are_spoken_plainly_with_the_reason_kept(fake):
    error = asyncio.run(fake("error").run("x"))
    assert not error.ok and "problem" in error.spoken and error.text == "API error"
    crash = asyncio.run(fake("crash").run("x"))
    assert not crash.ok and crash.text == "Traceback: boom"


def test_missing_claude_is_reported_not_raised(tmp_path):
    result = asyncio.run(Delegate(str(tmp_path), command=("no-such-claude-binary",)).run("x"))
    assert not result.ok and "couldn't find Claude Code" in result.spoken


def test_cancelling_stops_claude_instead_of_leaving_it_running(fake):
    async def go():
        task = asyncio.create_task(fake("slow").run("x"))
        await asyncio.sleep(0.5)
        task.cancel()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started

    assert asyncio.run(go()) < 5  # killed, not waited out for 30 s


def test_summary_falls_back_to_two_sentences_without_a_spoken_line():
    assert spoken_summary("Done. Tests pass. Also refactored a lot.") == "Done. Tests pass."
    assert spoken_summary("Notes\nSPOKEN: Fixed the bug.\n") == "Fixed the bug."
