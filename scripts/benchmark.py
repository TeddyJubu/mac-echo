"""MacEcho benchmark: where does a voice turn's time go?

    .venv/bin/python scripts/benchmark.py [--model REPO] [--turns N] [--out FILE]

No mic or speaker needed. src/macecho/vad/en.wav is fed through the real framing + VAD code
(Agent.process_audio); the speech segments it finds are replayed through the real
SpeechModel.stream(). Stages are timed by wrapping library calls, so the app is untouched.
Needs the GPU (Metal), so it can't run inside a sandbox.
"""

import time

T_START = time.perf_counter()

import argparse
import asyncio
import collections
import contextlib
import io
import json
import math
import os
import platform
import resource
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

T_IMPORTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)  # MacEchoConfig reads .env from the working directory

# Latency budget: metric name (see the "Headline" table) -> worst acceptable value.
# Metrics over budget are flagged OVER and the script exits 1, so it can guard regressions.
BUDGET: dict[str, float] = {
    # Thresholds leave room for background load, which moved first audio between 0.24 and 0.32 s.
    "perceived_latency_s": 0.95,  # 0.85-0.93 now; 1.05-1.08 when the VAD wait was 0.8 s
    "first_audio_s": 0.35,
    "cold_first_audio_s": 0.30,   # was 0.49 before the warm-up reply at load
    "underrun_ms": 0.0,           # playback must never run dry mid-reply
    "rtf": 0.5,                   # baseline 0.32; 1.0 is where playback stutters
    "startup_s": 3.0,             # 2.1 before; the load now includes a ~0.35 s warm-up reply
}

STAGES = {
    "prep": "prompt template + audio features",
    "prefill": "audio encoder + prompt prefill (MLX)",
    "text": "text token steps (MLX)",
    "audio": "speech frame steps (MLX)",
    "decode": "codec decode to waveform (MLX)",
    "glue": "everything else (Python glue)",
}
acc = collections.defaultdict(float)  # seconds per stage for the turn in progress
now, median = time.perf_counter, statistics.median


def timed(name, fn):
    def wrapper(*args, **kwargs):
        t = now()
        try:
            return fn(*args, **kwargs)
        finally:
            acc[name] += now() - t
    return wrapper


def bench_frontend(config):
    """Framing + VAD cost per 32 ms frame, and the speech segments VAD finds in en.wav."""
    from macecho.agent import Agent
    from macecho.vad.vad import VadConfig, VadProcessor

    with wave.open(str(ROOT / "src/macecho/vad/en.wav")) as w:
        pcm, seconds = w.readframes(w.getnframes()), w.getnframes() / w.getframerate()
    rate = config.audio_recording.sample_rate
    agent = object.__new__(Agent)  # skip __init__: no audio devices or speech model needed
    agent.config, agent.debug_audio_dir = config, None
    with contextlib.redirect_stdout(io.StringIO()):
        agent.vad = VadProcessor(VadConfig(
            threshold=config.vad.threshold, sampling_rate=rate,
            padding_duration=config.vad.padding_duration,
            min_speech_duration=config.vad.min_speech_duration,
            silence_duration=0.8,  # pinned: the same test utterances whatever the app's setting
            per_frame_duration=config.vad.per_frame_duration, model_path=config.vad.model_path))
    agent.frame_size = int(32.0 * rate / 1000)
    agent.audio_buffer = collections.deque(maxlen=agent.frame_size * 10)
    agent.vad.session.run = timed("onnx", agent.vad.session.run)
    segments = []

    async def collect(segment):
        segments.append(segment)
    agent._process_speech_segment = collect

    async def feed():
        step = config.audio_recording.chunk_size * 2  # int16 bytes per mic callback
        for i in range(0, len(pcm), step):
            await agent.process_audio(pcm[i:i + step])

    acc.clear()
    t = now()
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(feed())
    total, frames = now() - t, len(pcm) // 2 // agent.frame_size
    return {"frame_ms": 1000 * total / frames, "frame_onnx_ms": 1000 * acc["onnx"] / frames,
            "cpu_pct_of_realtime": 100 * total / seconds, "segments_found": len(segments)}, segments


def instrument(speech):
    """Time each stage of SpeechModel.stream() by wrapping the library calls it makes."""
    import mlx.core as mx
    import mlx_vlm.prompt_utils
    import mlx_vlm.utils

    mlx_vlm.prompt_utils.apply_chat_template = timed("prep", mlx_vlm.prompt_utils.apply_chat_template)
    mlx_vlm.utils.prepare_inputs = timed("prep", mlx_vlm.utils.prepare_inputs)
    codec, slots = speech.model.codec(), speech.model.stream_slots
    decode = codec.decode
    # MLX is lazy: force the result here, or the decode would be billed to whoever reads it
    object.__setattr__(codec, "decode", timed("decode", lambda *a, **k: mx.eval(r := decode(*a, **k)) or r))

    def stream_slots(*args, **kwargs):
        gen, stage = slots(*args, **kwargs), "prefill"
        while True:
            t = now()
            try:
                item = next(gen)
            except StopIteration:
                return
            # the first slot carries the prompt prefill; later ones are one model step each
            acc[stage if stage == "prefill" else "audio" if item[1] is not None else "text"] += now() - t
            stage = "step"
            yield item
    object.__setattr__(speech.model, "stream_slots", stream_slots)


def run_turn(speech, audio, **kwargs):
    """One reply, consumed the way Agent._process_speech_model consumes it."""
    import mlx.core as mx

    acc.clear()
    mx.random.seed(0)  # speech sampling is random; fix it so repeats time the same reply
    t0, cpu0 = now(), time.thread_time()
    gen, first, samples, buffered, underrun = speech.stream(audio, **kwargs), None, 0, 0.0, 0.0
    while True:
        try:
            chunk = next(gen)
        except StopIteration as done:
            text = done.value or ""
            break
        (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        if first is None:
            first = {"ttfa": now() - t0, **acc}
        # simulated speaker: it runs dry if this chunk lands after the audio queued so far ended
        underrun += max(0.0, now() - buffered) if samples else 0.0
        buffered = max(buffered, now()) + chunk.size / 24000
        samples += chunk.size
    # CPU time of this thread: the interpreter plus native CPU-side work. The rest of the
    # wall time the thread is blocked waiting for the GPU, which no rewrite shortens.
    turn = {"total": now() - t0, "cpu": time.thread_time() - cpu0, "underrun": underrun,
            "audio_s": samples / 24000, "reply": text, **acc}
    for d in filter(None, (turn, first)):
        d["glue"] = d.get("ttfa", d.get("total")) - sum(d.get(s, 0.0) for s in STAGES if s != "glue")
    return turn, first


def ranked(title, rows, total):
    print(f"\n{title}")
    for name, seconds in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<40}{1000 * seconds:8.0f} ms {100 * seconds / total:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="speech model repo (default: MACECHO_SPEECH_MODEL)")
    parser.add_argument("--turns", type=int, default=3, help="repeats per utterance")
    parser.add_argument("--utterances", type=int, default=3)
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()

    t = now()
    import mlx.core as mx
    import mlx_vlm  # noqa: F401
    import macecho.agent  # noqa: F401
    from macecho.config import MacEchoConfig
    from macecho.speech_model import SpeechModel
    import_s = T_IMPORTED - T_START + now() - t  # everything the app imports before loading weights
    config = MacEchoConfig()
    repo = args.model or config.speech_model or "LiquidAI/LFM3-Audio-2B-MLX-4bit"

    frontend, segments = bench_frontend(config)
    rate = config.audio_recording.sample_rate
    utterances = [s.astype(np.float32) / (32768.0 if s.dtype == np.int16 else 1.0)
                  for s in segments if 1.0 <= len(s) / rate <= 10.0][:args.utterances]
    assert utterances, "VAD found no 1-10 s speech segments in en.wav"

    t = now()
    speech = SpeechModel(repo)
    load_s = now() - t

    cold, cold_first = run_turn(speech, utterances[0])  # first turn after load: nothing is warm
    instrument(speech)
    turns = [run_turn(speech, u) for u in utterances for _ in range(args.turns)]
    turns = [(turn, first) for turn, first in turns if first]  # drop misfires (no speech out)
    assert turns, "the model produced no speech for any utterance"
    t = now()
    speech.transcribe(utterances[0])
    transcribe_s = now() - t

    def mid(key, which):
        return median(pair[which].get(key, 0.0) for pair in turns)

    # the VAD counts silence in whole 32 ms frames, so the real wait rounds up
    frame = config.vad.per_frame_duration
    ttfa, wait = mid("ttfa", 1), math.ceil(round(config.vad.silence_duration / frame, 6)) * frame
    headline = {
        "startup_s": import_s + load_s,
        "cold_first_audio_s": cold_first["ttfa"] if cold_first else float("nan"),
        "first_audio_s": ttfa,
        "perceived_latency_s": wait + ttfa,
        "underrun_ms": 1000 * max(turn["underrun"] for turn, _ in turns),
        "rtf": median(turn["total"] / turn["audio_s"] for turn, _ in turns),
        "memory_transcribe_s": transcribe_s,
        "frontend_cpu_pct": frontend["cpu_pct_of_realtime"],
        "python_thread_cpu_pct": 100 * median(turn["cpu"] / turn["total"] for turn, _ in turns),
        "peak_gpu_memory_gb": mx.get_peak_memory() / 1e9,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
    }
    results = {
        "model": repo, "machine": platform.platform(), "python": platform.python_version(),
        "turns_measured": len(turns), "headline": headline, "startup": {"import_s": import_s, "load_s": load_s},
        "frontend": frontend,
        "first_audio_stages_s": {s: mid(s, 1) for s in STAGES},
        "turn_stages_s": {s: mid(s, 0) for s in STAGES},
        "turn": {"total_s": mid("total", 0), "audio_s": mid("audio_s", 0),
                 "ttfa_min_s": min(f["ttfa"] for _, f in turns), "ttfa_max_s": max(f["ttfa"] for _, f in turns)},
        "replies": sorted({turn["reply"] for turn, _ in turns}),
    }

    print(f"\nMacEcho benchmark — {repo}\n{results['machine']}, {len(turns)} warm turns")
    print("\nHeadline")
    over = []
    for name, value in headline.items():
        flag = "  OVER" if name in BUDGET and value > BUDGET[name] else ""
        over += [name] if flag else []
        print(f"  {name:<40}{value:8.2f}{flag}")
    ranked("Perceived latency: user stops talking -> first audio",
           {"VAD end-of-speech wait (config)": wait,
            **{STAGES[s]: results["first_audio_stages_s"][s] for s in STAGES}}, wait + ttfa)
    ranked(f"Whole turn ({results['turn']['audio_s']:.1f} s of speech out)",
           {STAGES[s]: results["turn_stages_s"][s] for s in STAGES}, results["turn"]["total_s"])
    print(f"\nFront end: {frontend['frame_ms']:.3f} ms per 32 ms frame "
          f"({frontend['frame_onnx_ms']:.3f} ms of it in ONNX Runtime)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nSaved {args.out}")
    sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
