"""The macecho agent CLI: protocol, daemon server, client lifecycle, listening, voice, skill.

Nothing here needs the GPU, the mic or the speaker: the server runs with a fake engine, and the
listening logic gets the real VAD fed from a WAV file.
"""
import json
import socket
import tempfile
import threading
import time

import pytest

from macecho.daemon import protocol, server


class FakeEngine:
    def __init__(self):
        self.calls, self.fail = [], None

    def listen(self, timeout):
        self.calls.append(("listen", timeout))
        if self.fail:
            raise server.EngineError(self.fail, "no speech")
        return {"text": "hello there", "seconds": 1.5}

    def say(self, text):
        self.calls.append(("say", text))

    def status(self):
        return {"model": "fake", "voice": "system"}


@pytest.fixture
def runtime(monkeypatch):
    # A short directory: Unix socket paths must stay under 104 bytes.
    path = tempfile.mkdtemp(prefix="me")
    monkeypatch.setenv("MACECHO_RUNTIME_DIR", path)
    return path


def start(engine, **kwargs):
    thread = threading.Thread(target=server.serve, args=(engine,), kwargs={"poll": 0.02, **kwargs},
                              daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not protocol.socket_path().exists():
        assert time.time() < deadline, "server never bound its socket"
        time.sleep(0.01)
    return thread


def ask(obj):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.connect(str(protocol.socket_path()))
        protocol.send(sock, obj)
        return protocol.recv(sock)


def test_each_op_round_trips_and_converse_says_then_listens(runtime):
    engine = FakeEngine()
    thread = start(engine, idle_minutes=0)
    assert ask({"op": "listen", "timeout": 5}) == {"ok": True, "text": "hello there", "seconds": 1.5}
    assert ask({"op": "say", "text": "hi"}) == {"ok": True}
    assert ask({"op": "converse", "text": "ready?", "timeout": 7})["text"] == "hello there"
    status = ask({"op": "status"})
    assert status["ok"] and status["model"] == "fake" and status["pid"] > 0
    assert engine.calls == [("listen", 5.0), ("say", "hi"), ("say", "ready?"), ("listen", 7.0)]
    assert ask({"op": "stop"}) == {"ok": True}
    thread.join(5)
    assert not thread.is_alive() and not protocol.socket_path().exists()


def test_engine_errors_keep_their_kind_and_a_long_text_arrives_whole(runtime):
    engine = FakeEngine()
    start(engine, idle_minutes=0)
    engine.fail = "timeout"
    assert ask({"op": "listen", "timeout": 1}) == {"ok": False, "error": "timeout", "message": "no speech"}
    long_text = "word " * 4000  # 20 000 characters: more than one socket read
    assert ask({"op": "say", "text": long_text}) == {"ok": True}
    assert engine.calls[-1] == ("say", long_text)
    ask({"op": "stop"})


def test_bad_requests_and_vanished_clients_do_not_wedge_the_server(runtime):
    start(FakeEngine(), idle_minutes=0)
    with socket.socket(socket.AF_UNIX) as sock:
        sock.connect(str(protocol.socket_path()))
        sock.sendall(b"\xff not json\n")
        reply = protocol.recv(sock)
    assert reply["ok"] is False and reply["error"] == "internal"
    with socket.socket(socket.AF_UNIX) as sock:  # a client killed before reading its reply
        sock.connect(str(protocol.socket_path()))
        protocol.send(sock, {"op": "listen", "timeout": 1})
    assert ask({"op": "say", "text": "still here"}) == {"ok": True}
    ask({"op": "stop"})


def test_idle_daemon_exits_and_removes_its_socket(runtime):
    now = [0.0]
    thread = start(FakeEngine(), idle_minutes=1, clock=lambda: now[0])
    now[0] = 61.0  # a minute and a second with no request
    thread.join(5)
    assert not thread.is_alive() and not protocol.socket_path().exists()


def test_only_one_daemon_serves_at_a_time(runtime):
    start(FakeEngine(), idle_minutes=0)
    assert server.serve(FakeEngine(), idle_minutes=0, poll=0.02) is False  # lock already held
    assert ask({"op": "say", "text": "first one"}) == {"ok": True}
    ask({"op": "stop"})
