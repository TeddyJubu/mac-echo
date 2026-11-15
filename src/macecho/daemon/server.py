"""The daemon's socket server: one request at a time, in front of an engine.

The engine does the real work (`listen(timeout) -> dict`, `say(text)`, `status() -> dict`) and
raises EngineError for failures the client should report with a specific exit code. Requests are
handled one after another on one thread, so two agents calling at once take turns for the mic.

An engine with a `live` attribute (engine.Live) also gets a second socket, live.sock, served on
its own threads: a window connects once and reads a JSON line about every 50 ms, even while a
listen or say is holding the main thread.
"""
import fcntl
import os
import socket
import threading
import time

from macecho.daemon import protocol


class EngineError(Exception):
    def __init__(self, kind: str, message: str = ""):
        super().__init__(message)
        self.kind = kind


def handle(engine, request: dict) -> dict:
    op = request.get("op")
    try:
        if op == "listen":
            return {"ok": True, **engine.listen(float(request.get("timeout", 120)))}
        if op == "say":
            engine.say(str(request["text"]))
            return {"ok": True}
        if op == "converse":
            engine.say(str(request["text"]))
            return {"ok": True, **engine.listen(float(request.get("timeout", 120)))}
        if op == "status":
            return {"ok": True, "pid": os.getpid(), **engine.status()}
        return {"ok": False, "error": "internal", "message": f"unknown op {op!r}"}
    except EngineError as e:
        return {"ok": False, "error": e.kind, "message": str(e)}
    except Exception as e:  # report it to the client; the daemon keeps serving
        return {"ok": False, "error": "internal", "message": f"{type(e).__name__}: {e}"}


def serve(engine, idle_minutes: float = 30, clock=time.monotonic, poll: float = 1.0) -> bool:
    """Serve until `stop` or `idle_minutes` without a request (0 never idles out).

    `engine` is an engine, or a function that builds one: it is called only once the lock is
    held, so a second daemon started in a race exits before loading the model. Returns False at
    once if another daemon holds the lock: the caller's clients reach that one.
    """
    protocol.runtime_dir().mkdir(parents=True, exist_ok=True)
    lock = open(protocol.lock_path(), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return False
    if callable(engine):
        engine = engine()
    path = protocol.socket_path()
    path.unlink(missing_ok=True)  # left by a daemon that crashed; we hold the lock, so it is stale
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen()
    listener.settimeout(poll)  # wake up to check the idle deadline
    live = _open_live(engine.live) if hasattr(engine, "live") else None
    last = clock()
    try:
        while not (idle_minutes and clock() - last > idle_minutes * 60):
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(None)
                if _serve_one(engine, conn, clock() - last, idle_minutes) == "stop":
                    return True
            last = clock()
    finally:
        listener.close()
        path.unlink(missing_ok=True)
        if live:
            live.close()  # its accept thread sees the closed socket within 0.2 s and ends
            protocol.live_path().unlink(missing_ok=True)
        lock.close()
    return True


def _open_live(state, every: float = 0.05):
    """Bind live.sock and stream state.snapshot() to each window that connects."""
    path = protocol.live_path()
    path.unlink(missing_ok=True)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen()
    # close() from another thread doesn't reliably wake a blocked accept(); a timeout does.
    listener.settimeout(0.2)

    def stream(conn):
        with conn:
            try:
                while True:
                    protocol.send(conn, state.snapshot())
                    time.sleep(every)
            except OSError:
                pass  # the window closed

    def accept():
        while listener.fileno() != -1:  # -1 once the daemon closed it: it is stopping
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            # ponytail: one thread per window; there is only ever one or two
            threading.Thread(target=stream, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return listener


def _serve_one(engine, conn, idle_seconds, idle_minutes):
    try:
        request = protocol.recv(conn)
    except ValueError as e:
        reply = {"ok": False, "error": "internal", "message": f"bad request: {e}"}
    else:
        if request is None:
            return None  # the client hung up before asking anything
        if request.get("op") == "stop":
            _reply(conn, {"ok": True})
            return "stop"
        # ponytail: a client killed mid-listen still waits out its listen here; check the
        # socket while listening if agents abort often.
        reply = handle(engine, request)
        if request.get("op") == "status" and reply["ok"]:
            reply["idle_minutes_left"] = round(idle_minutes - idle_seconds / 60, 1) if idle_minutes else None
    _reply(conn, reply)
    return None


def _reply(conn, reply):
    try:
        protocol.send(conn, reply)
    except OSError:
        pass  # the client went away (Ctrl-C); the next request is unaffected
