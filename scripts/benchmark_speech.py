"""Speech benchmark: how fast MacEcho hears and speaks, and whether each understands the other.

    .venv/bin/python scripts/benchmark_speech.py [--voice NAME ...] [--out FILE]

ZONOS3-Mini reads each sentence aloud (timed to its first chunk), Fluid Speech Pro transcribes the
result (timed), and the words must come back exactly. That round trip catches a voice that babbles,
skips or garbles words, and ears that mishear. Needs Metal, so no sandbox.

Baseline: benchmarks/speech-baseline.json. Compare runs made back to back; both swing with load.
"""
import argparse
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soxr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macecho.speech_model import DEFAULT_VOICE, RATE, SpeechModel  # noqa: E402

SENTENCES = [
    "Tell me a story about a cat.",
    "What does the greet function in app.py return?",
    "Never mind, stop that.",
    "Once upon a time, a curious orange cat named Marmalade found a loose floorboard.",
    "Commit my changes and push them.",
    "Which branch am I on?",
]

# Metric -> worst acceptable value; anything over is flagged OVER and the script exits 1.
BUDGET = {
    "first_audio_p50_s": 0.35,     # a sentence handed to the voice -> its first sound
    "transcribe_p50_ms": 250,      # 2-8 s of speech -> text (LFM3 took 260-950 ms)
    "round_trip_wrong": 0,         # sentences that didn't come back word for word
}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", nargs="+", default=[DEFAULT_VOICE], help=f"voices to try ({DEFAULT_VOICE})")
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()

    started = time.perf_counter()
    speech = SpeechModel(args.voice[0])  # warms both models up, as the app does
    print(f"loaded and warmed up in {time.perf_counter() - started:.1f}s", flush=True)
    # Loaded on this thread, used on another: the app loads on one thread and speaks and hears from
    # asyncio's worker threads, and MLX streams belong to the thread that made them.
    worker = ThreadPoolExecutor(1)
    rows = []
    for voice in args.voice:
        speech.voice = voice
        for sentence in SENTENCES:
            start, first, chunks = time.perf_counter(), None, []
            def speak():
                nonlocal first
                for chunk in speech.tts(sentence):
                    first = first if first is not None else time.perf_counter() - start
                    chunks.append(chunk)
            worker.submit(speak).result()
            audio = np.concatenate(chunks)
            start = time.perf_counter()
            heard = worker.submit(speech.transcribe, soxr.resample(audio, RATE, 16000).astype(np.float32)).result()
            hear_ms = (time.perf_counter() - start) * 1000
            ok = words(heard) == words(sentence)
            rows.append({"voice": voice, "said": sentence, "heard": heard, "ok": ok, "first_audio_s": round(first, 3),
                         "speech_s": round(audio.size / RATE, 2), "transcribe_ms": round(hear_ms)})
            print(f"  {'ok ' if ok else 'BAD'} [{voice}] first audio {first:.2f}s, {audio.size / RATE:.1f}s of speech, "
                  f"heard in {hear_ms:.0f} ms: {heard!r}", flush=True)

    metrics = {
        "first_audio_p50_s": round(statistics.median(r["first_audio_s"] for r in rows), 3),
        "transcribe_p50_ms": round(statistics.median(r["transcribe_ms"] for r in rows)),
        "round_trip_wrong": sum(not r["ok"] for r in rows),
    }
    print("\nMetric                     value   budget")
    over = []
    for name, value in metrics.items():
        bad = value > BUDGET[name]
        over += [name] if bad else []
        print(f"  {name:22} {value!s:>8}  {BUDGET[name]!s:>6}  {'OVER' if bad else 'ok'}")
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": metrics, "budget": BUDGET, "rows": rows}, indent=2))
    sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
