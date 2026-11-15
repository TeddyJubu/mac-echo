"""Orb benchmark: how far behind the daemon can the window be, and what does feeding it cost?

    .venv/bin/python scripts/benchmark_orb.py [--changes N] [--out FILE]

No model, mic or GPU: a real daemon server with a real Live state, and the real ViewerApp reader
thread on the other end of live.sock. Each state change is timed until the viewer has it. The
viewer then pushes to the page on a 50 ms timer, so the page lags by at most that much more.
Binds Unix sockets, so it can't run inside a sandbox.

Baseline: benchmarks/orb-baseline.json.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MACECHO_RUNTIME_DIR"] = tempfile.mkdtemp(prefix="me")  # short: socket path limit

from macecho import menubar  # noqa: E402
from macecho.daemon import protocol, server  # noqa: E402
from macecho.daemon.engine import Live  # noqa: E402

PAGE_TIMER_MS = 50  # ViewerApp pushes to the page every 0.05 s (rumps.Timer in ViewerApp.__init__)

# Metric -> worst acceptable value; anything over is flagged OVER and the script exits 1.
BUDGET = {
    "state_to_viewer_p95_ms": 70,    # the stream sends every 50 ms; more means it falls behind
    "state_to_page_worst_ms": 130,   # what the user can see: viewer max + the page timer
    "snapshot_ms": 1.0,              # one live line: levels + 13-band FFT, in the speech process
    "daemon_core_pct": 2.0,          # that, 20 times a second, as a share of one core
}


class Engine:
    """Just the live state: the server's request side is never used here."""
    live = Live()

    def status(self):
        return {}


def time_state_changes(changes: int) -> list[float]:
    engine = Engine()
    threading.Thread(target=server.serve, args=(engine,), kwargs={"idle_minutes": 0, "poll": 0.05},
                     daemon=True).start()
    viewer = object.__new__(menubar.ViewerApp)  # no Cocoa: only its reader thread
    viewer.live = None
    threading.Thread(target=viewer._read_live, kwargs={"retry": 0.05}, daemon=True).start()
    deadline = time.monotonic() + 5
    while viewer.live is None:
        assert time.monotonic() < deadline, "the viewer never connected to live.sock"
        time.sleep(0.005)

    delays = []
    for i in range(changes):
        state = ("listening", "replying")[i % 2]
        time.sleep(np.random.uniform(0, 0.05))  # land anywhere in the stream's 50 ms cycle
        start = time.perf_counter()
        engine.live.state = state
        while viewer.state() != state:
            time.sleep(0.0005)
        delays.append((time.perf_counter() - start) * 1000)
    return delays


def time_snapshot(calls: int = 2000) -> float:
    live = Live()
    live.state = "replying"
    live.heard = (np.random.randn(512) * 3000).astype(np.int16)  # one 32 ms mic frame
    t = np.arange(int(24000 * 0.32)) / 24000
    live.saying = ((0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), time.monotonic(), 24000)
    start = time.perf_counter()
    for _ in range(calls):
        json.dumps(live.snapshot())  # what each line costs to build and encode
    return (time.perf_counter() - start) / calls * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--changes", type=int, default=100, help="state changes to time (100)")
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()

    delays = time_state_changes(args.changes)
    snapshot_ms = time_snapshot()
    quantiles = statistics.quantiles(delays, n=20)  # 5 % steps: [9] is the median, [18] p95
    metrics = {
        "state_to_viewer_p50_ms": round(quantiles[9], 1),
        "state_to_viewer_p95_ms": round(quantiles[18], 1),
        "state_to_viewer_max_ms": round(max(delays), 1),
        "state_to_page_worst_ms": round(max(delays) + PAGE_TIMER_MS, 1),
        "snapshot_ms": round(snapshot_ms, 3),
        "daemon_core_pct": round(snapshot_ms * 20 / 10, 2),  # ms per s at 20 Hz -> % of one core
    }

    print(f"{args.changes} state changes through live.sock to the viewer:")
    print("\nMetric                        value    budget")
    over = []
    for name, value in metrics.items():
        limit = BUDGET.get(name)
        flag = "" if limit is None else ("OVER" if value > limit else "ok")
        over += [name] if flag == "OVER" else []
        print(f"  {name:26} {value!s:>8}  {'' if limit is None else limit!s:>7}  {flag}")
    if args.out:
        Path(args.out).write_text(json.dumps({"metrics": metrics, "budget": BUDGET,
                                              "delays_ms": [round(d, 2) for d in delays]}, indent=2))
    protocol.live_path().unlink(missing_ok=True)
    sys.exit(1 if over else 0)


if __name__ == "__main__":
    main()
