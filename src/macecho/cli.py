"""`macecho`: ears and a voice for AI agents.

    macecho listen   [--timeout SECS] [--json]   wait for one spoken sentence, print it
    macecho say      "text"                      speak (or pipe the text on stdin)
    macecho converse "text" [--timeout] [--json] say, then listen: one agent turn, one call
    macecho status | stop
    macecho show                                  open the window that shows listening and speaking
    macecho guide                                 how an agent should hold the conversation
    macecho install-skill [--force]               add the Claude Code skill (done automatically)

stdout carries only the result. Exit codes: 0 ok, 1 error, 2 nobody spoke before the timeout,
3 microphone access is off. The first command starts a background daemon that keeps the speech
model loaded (see macecho/daemon); it stops itself after MACECHO_IDLE_MINUTES (30) unused.

Standard library only: this runs on every agent turn and must start instantly.
"""
import argparse
import fcntl
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from macecho.daemon import protocol

GUIDE = Path(__file__).with_name("agent_guide.md")
SKILL_HEADER = """---
name: macecho-voice
description: Talk with the user by voice on their Mac through the macecho CLI. Use when the user says "voice mode", "let's talk" or "talk to me", or asks you to listen to them or answer out loud.
---

"""
MIC_HELP = ("Microphone access is off for the app running this command (your terminal or editor). "
            "Turn it on in System Settings › Privacy & Security › Microphone, then try again.")


def skill_path() -> Path:
    return Path.home() / ".claude/skills/macecho-voice/SKILL.md"


def install_skill() -> Path:
    path = skill_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SKILL_HEADER + GUIDE.read_text())
    return path


def _auto_install_skill() -> None:
    """First run with Claude Code present: add the skill, once, and never over the user's copy."""
    if os.environ.get("MACECHO_NO_SKILL") or not (Path.home() / ".claude").is_dir() or skill_path().exists():
        return
    try:
        install_skill()
    except OSError:
        return  # a nicety; never let it break the command the agent actually ran
    print("Installed Claude Code skill: ~/.claude/skills/macecho-voice "
          "(set MACECHO_NO_SKILL=1 to skip)", file=sys.stderr)


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


def _viewer_open() -> bool:
    """The window holds viewer.lock while it is open (menubar.take_viewer_lock)."""
    try:
        with open(protocol.viewer_lock_path(), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
    except BlockingIOError:
        return True
    except FileNotFoundError:
        return False


def show(spawn=None) -> int:
    """Open the window that draws the daemon's listening and speaking; return at once."""
    if _viewer_open():
        return 0
    # find_spec locates without importing, so this stays instant; the window needs the app extras.
    if not all(importlib.util.find_spec(name) for name in ("rumps", "WebKit")):
        # An installed tool's __file__ is in its own site-packages, so the checkout isn't known here.
        print("macecho: the window needs the app extras: "
              "uv tool install --force '/path/to/mac-echo[app]'", file=sys.stderr)
        return 1
    if spawn:
        spawn()
        return 0
    log = protocol.app_log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        # ponytail: shows as "Python" in the Dock; launching dist/MacEcho.app would show its icon.
        subprocess.Popen([sys.executable, "-m", "macecho.menubar", "--view"], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return 0


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
    sub.add_parser("show", help="open the window that shows listening and speaking")
    sub.add_parser("guide", help="print how an agent should hold a voice conversation")
    sub.add_parser("install-skill", help="install the Claude Code skill").add_argument(
        "--force", action="store_true", help="overwrite an existing copy")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "install-skill":
        if skill_path().exists() and not args.force:
            print(f"Already installed: {skill_path()} (--force to overwrite)", file=sys.stderr)
        else:
            print(install_skill())
        return 0
    _auto_install_skill()
    if args.command == "show":
        return show()
    if args.command == "guide":
        print(GUIDE.read_text(), end="")
        return 0
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
