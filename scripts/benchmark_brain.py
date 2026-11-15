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
from macecho.brain import MODEL, Brain, Stop, Task  # noqa: E402

# Metric -> limit. accuracy is a floor; the rest are ceilings. Anything out flags OUT and exits 1.
BUDGET = {
    "accuracy": 0.95,           # floor: LFM3's best was 0.78
    "first_sound_p50_s": 0.8,   # request sent -> first sentence (or hand-over) ready to speak
    "first_sound_p95_s": 1.5,
    "dialogues_wrong": 0,
}

# Conversations that once went wrong, replayed turn by turn: what the last turn must do.
DIALOGUES = [
    # 2026-11-15: "tell me a story" misheard; then searched the project for a story it should write
    (["My story about a cat.", "Make it up."], "CHAT"),
    # 2026-11-15: answered "Okay, I've stopped that." with no way to stop; Claude kept going
    (["Find my cat story in the project folder.", "No, stop that."], "STOP"),
    (["Write me a short poem about the sea."], "CHAT"),
]


async def one(client, model, text):
    start, first, got = time.perf_counter(), None, "CHAT"
    async for item in Brain(client, model).reply(text):
        if first is None:
            first = time.perf_counter() - start
        if isinstance(item, Task):
            got = "CLAUDE"
    return got, first if first is not None else time.perf_counter() - start


async def dialogue(client, model, turns):
    """What the last turn did: STOP, CLAUDE or CHAT (earlier turns run too, so it has context)."""
    brain = Brain(client, model)
    for turn in turns[:-1]:
        async for _ in brain.reply(turn):
            pass
    items = [item async for item in brain.reply(turns[-1])]
    return ("STOP" if any(isinstance(i, Stop) for i in items) else
            "CLAUDE" if any(isinstance(i, Task) for i in items) else "CHAT")


async def run(model):
    client, rows = Brain(model=model).client, []  # the app's own client set-up, shared by every case
    for text, expected in CASES:
        got, first = await one(client, model, text)
        rows.append({"text": text, "expected": expected, "got": got, "first_s": round(first, 3)})
        print(f"  {'ok ' if got == expected else 'BAD'} {first:5.2f}s  {expected:6} -> {got:6}  {text}", flush=True)
    for turns, expected in DIALOGUES:
        got = await dialogue(client, model, turns)
        rows.append({"dialogue": turns, "expected": expected, "got": got})
        print(f"  {'ok ' if got == expected else 'BAD'} dialogue {expected:6} -> {got:6}  {' / '.join(turns)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()

    rows = asyncio.run(run(args.model))
    talks = [r for r in rows if "dialogue" in r]
    rows = [r for r in rows if "dialogue" not in r]
    quantiles = statistics.quantiles([r["first_s"] for r in rows], n=20)  # [9] median, [18] p95
    metrics = {
        "accuracy": round(sum(r["got"] == r["expected"] for r in rows) / len(rows), 3),
        "chat_sent_to_claude": sum(r["expected"] == "CHAT" and r["got"] == "CLAUDE" for r in rows),
        "claude_kept_as_chat": sum(r["expected"] == "CLAUDE" and r["got"] == "CHAT" for r in rows),
        "first_sound_p50_s": round(quantiles[9], 3),
        "first_sound_p95_s": round(quantiles[18], 3),
        "dialogues_wrong": sum(t["got"] != t["expected"] for t in talks),
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
                                              "rows": rows, "dialogues": talks}, indent=2))
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
