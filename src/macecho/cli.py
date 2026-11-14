"""`macecho`: ears and a voice for AI agents.

    macecho listen   [--timeout SECS] [--json]   wait for one spoken sentence, print it
    macecho say      "text"                      speak (or pipe the text on stdin)
    macecho converse "text" [--timeout] [--json] say, then listen: one agent turn, one call
    macecho status | stop

stdout carries only the result. Exit codes: 0 ok, 1 error, 2 nobody spoke before the timeout,
3 microphone access is off. The first command starts a background daemon that keeps the speech
model loaded (see macecho/daemon); it stops itself after MACECHO_IDLE_MINUTES (30) unused.

Standard library only: this runs on every agent turn and must start instantly.
"""
import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from macecho.daemon import protocol

MIC_HELP = ("Microphone access is off for the app running this command (your terminal or editor). "
            "Turn it on in System Settings › Privacy & Security › Microphone, then try again.")


def _connect() -> socket.socket | None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(protocol.socket_path()))
        return sock
    except OSError:  # missing, refused, or not a socket (a stale file)
        sock.close()
        return None


def _daemon_running() -> bool:
    """A live daemon holds the lock from before it binds until after it unlinks the socket."""
    try:
        with open(protocol.lock_path(), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
    except BlockingIOError:
        return True
    except FileNotFoundError:  # no runtime dir yet: nothing has ever run
        return False


def _spawn_daemon() -> None:
    log = protocol.log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        subprocess.Popen([sys.executable, "-m", "macecho.daemon"], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)


def _model_cached(repo: str) -> bool:
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface") / "hub")
    return (hub / ("models--" + repo.replace("/", "--"))).exists()


def _start_daemon(spawn, wait: float) -> socket.socket | None:
    """Start the daemon if none is starting, then wait for its socket."""
    if not _daemon_running():
        protocol.socket_path().unlink(missing_ok=True)  # a crashed daemon's leftover
        repo = os.environ.get("MACECHO_SPEECH_MODEL", protocol.DEFAULT_MODEL)
        print("Loading voice model…" if _model_cached(repo) else
              "Downloading voice model (first run only, ~1–2 GB)…", file=sys.stderr, flush=True)
        spawn()
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if sock := _connect():
            return sock
        time.sleep(0.1)
    return None


def request(obj: dict, spawn=None, wait: float = 600) -> dict:
    """Send one request, starting the daemon if needed; retry once if it dies mid-request."""
    spawn = spawn or _spawn_daemon
    for _ in range(2):
        sock = _connect() or _start_daemon(spawn, wait)
        if sock is None:
            break
        try:
            protocol.send(sock, obj)
            if (reply := protocol.recv(sock)) is not None:
                return reply
        except (OSError, ValueError):
            pass  # the daemon crashed or closed on us: go round once more
        finally:
            sock.close()
    return {"ok": False, "error": "internal",
            "message": f"the voice daemon stopped unexpectedly; see {protocol.log_path()}"}


def exit_code(reply: dict) -> int:
    return 0 if reply.get("ok") else protocol.ERRORS.get(reply.get("error"), 1)


def _report(reply: dict, as_json: bool) -> int:
    code = exit_code(reply)
    if code == 0 and "text" in reply:
        print(json.dumps({"text": reply["text"], "seconds": reply["seconds"]}) if as_json
              else reply["text"])
    elif code == 3:
        print(f"macecho: {MIC_HELP}", file=sys.stderr)
    elif code == 1:
        print(f"macecho: {reply.get('message', 'failed')}", file=sys.stderr)
    return code  # 2 (no speech) prints nothing: the agent just listens again or moves on


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="macecho", description="Ears and a voice for AI agents, on your Mac.",
                                epilog="Agents: run `macecho guide` for how to hold a conversation.")
    sub = p.add_subparsers(dest="command", required=True)
    for name, helptext in (("listen", "wait for one spoken sentence and print it"),
                           ("converse", "say TEXT, then listen and print the reply")):
        c = sub.add_parser(name, help=helptext)
        if name == "converse":
            c.add_argument("text")
        c.add_argument("--timeout", type=float, default=120, help="seconds to wait for speech (120)")
        c.add_argument("--json", action="store_true", help='print {"text", "seconds"}')
    sub.add_parser("say", help="speak TEXT (or stdin)").add_argument("text", nargs="?")
    sub.add_parser("status", help="show the daemon's state as JSON")
    sub.add_parser("stop", help="stop the daemon and free the model's memory")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "status":
        if not (sock := _connect()):
            print(json.dumps({"running": False}))
            return 0
        sock.close()
        reply = request({"op": "status"})
        print(json.dumps({"running": True, **{k: v for k, v in reply.items() if k != "ok"}}))
        return exit_code(reply)
    if args.command == "stop":
        if sock := _connect():
            sock.close()
            request({"op": "stop"})
        return 0
    if args.command == "say":
        text = args.text if args.text is not None else ("" if sys.stdin.isatty() else sys.stdin.read())
        if not text.strip():
            print("macecho: nothing to say", file=sys.stderr)
            return 1
        return _report(request({"op": "say", "text": text}), False)
    if args.command == "converse":
        return _report(request({"op": "converse", "text": args.text, "timeout": args.timeout}), args.json)
    return _report(request({"op": "listen", "timeout": args.timeout}), args.json)


if __name__ == "__main__":
    sys.exit(main())
