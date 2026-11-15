"""Delegation benchmark: a spoken task goes to Claude Code, and a short summary comes back aloud.

    .venv/bin/python scripts/benchmark_delegate.py [--no-voice] [--no-claude] [--voice NAME] [--out FILE]

Baseline: benchmarks/delegate-baseline.json. Compare runs made back to back; both halves swing with load.

Voice (needs Metal, so no sandbox): ZONOS3-Mini reading the acknowledgement and each summary; the number
that matters is how soon the first sound comes after the user stops talking.
Claude (needs the network and spends money, about $0.5-1 per task): three tasks in a throwaway git
repo, run through the real Delegate. Each is checked by its effect on the repo, not just its exit
code, and the third ("add a test for it") only works if the session carried over.
"""
import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macecho.delegate import Delegate  # noqa: E402

ACK = "Okay, Claude Code is working on it."

# Metric -> worst acceptable value; anything over is flagged OVER and the script exits 1.
# Claude's own time and cost depend on the network and the model, so they are reported, not judged.
BUDGET = {
    "ack_first_audio_s": 0.5,     # silence between "do X" and "okay, on it"
    "summary_first_audio_s": 0.5,
    "tasks_failed": 0,
    "long_reports": 0,            # over 100 words: Claude ignored the briefing
    "summary_words_max": 40,      # about 15 s of speech; longer belongs on screen
}

TASKS = [  # request, check(repo) -> did the work really happen
    ("What does this project do? Answer in one sentence.", lambda repo: True),
    ("Add a subtract(a, b) function to calc.py.",
     lambda repo: "def subtract" in (repo / "calc.py").read_text()),
    ("Now add a test for it to test_calc.py.",  # "it" only resolves if the session carried over
     lambda repo: "subtract" in (repo / "test_calc.py").read_text()),
]


def make_repo() -> Path:
    repo = Path(tempfile.mkdtemp(prefix="macecho-bench-"))
    (repo / "calc.py").write_text('"""A tiny calculator."""\n\n\ndef add(a, b):\n    return a + b\n')
    (repo / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "-c", "user.name=bench", "-c", "user.email=bench@localhost", "commit", "-qm", "init"]):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


def time_tts(speech, text: str) -> dict:
    """Seconds to the first chunk and to the end, and how long the speech itself lasts. Spoken a
    sentence at a time, as the voices do (daemon/voice.py), so the first sound waits for the first
    sentence only: one call for a whole summary took 1.1 s to start, against 0.2-0.4 s per sentence."""
    from macecho.daemon.voice import sentences, speakable

    start, first, samples = time.perf_counter(), None, 0
    for sentence in sentences(speakable(text)):
        for chunk in speech.tts(sentence):
            first = first if first is not None else time.perf_counter() - start
            samples += chunk.size
    return {"first_audio_s": round(first or 0.0, 3), "total_s": round(time.perf_counter() - start, 3),
            "audio_s": round(samples / 24000, 2)}


async def run_claude(repo: Path) -> list[dict]:
    delegate, rows = Delegate(str(repo)), []
    for request, check in TASKS:
        result = await delegate.run(request)
        rows.append({"request": request, "ok": result.ok, "done": result.ok and check(repo),
                     "seconds": round(result.seconds, 1), "cost_usd": round(result.cost_usd, 3),
                     "report_words": len(result.text.split()),
                     "summary": result.spoken, "summary_words": len(result.spoken.split()),
                     "session_id": result.session_id})
        print(f"  {'done' if rows[-1]['done'] else 'FAIL'}  {rows[-1]['seconds']:5.1f}s  "
              f"${rows[-1]['cost_usd']:.2f}  {request}\n        says: {result.spoken!r}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-voice", action="store_true", help="skip the voice timings")
    parser.add_argument("--no-claude", action="store_true", help="skip Claude Code (free, offline)")
    parser.add_argument("--voice", help="ZONOS3 voice (default: MACECHO_VOICE)")
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()
    results, metrics = {}, {}

    speech = None
    if not args.no_voice:
        from macecho.config import MacEchoConfig
        from macecho.speech_model import SpeechModel
        name = args.voice or MacEchoConfig().voice
        print(f"Loading Fluid Speech Pro and ZONOS3-Mini (voice: {name})...", flush=True)
        speech = SpeechModel(name)
        results["ack"] = [time_tts(speech, ACK) for _ in range(3)]
        metrics["ack_first_audio_s"] = max(r["first_audio_s"] for r in results["ack"])
        print(f"  ack: {results['ack']}")

    if not args.no_claude:
        repo = make_repo()
        print(f"Claude Code in {repo}:", flush=True)
        tasks = results["tasks"] = asyncio.run(run_claude(repo))
        metrics["tasks_failed"] = sum(not t["done"] for t in tasks)
        metrics["long_reports"] = sum(t["report_words"] > 100 for t in tasks)
        metrics["summary_words_max"] = max(t["summary_words"] for t in tasks)
        metrics["claude_s_median"] = sorted(t["seconds"] for t in tasks)[len(tasks) // 2]
        metrics["claude_cost_usd_total"] = round(sum(t["cost_usd"] for t in tasks), 2)
        metrics["same_session"] = len({t["session_id"] for t in tasks}) == 1
        if speech:
            results["summary_tts"] = [time_tts(speech, t["summary"]) for t in tasks if t["summary"]]
            metrics["summary_first_audio_s"] = max(r["first_audio_s"] for r in results["summary_tts"])

    print("\nMetric                        value    budget")
    over = []
    for name, value in metrics.items():
        limit = BUDGET.get(name)
        flag = "" if limit is None else ("OVER" if value > limit else "ok")
        over += [name] if flag == "OVER" else []
        print(f"  {name:26} {value!s:>8}  {'' if limit is None else limit!s:>7}  {flag}")
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": metrics, "budget": BUDGET, **results}, indent=2))
    sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
