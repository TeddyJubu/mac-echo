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

from macecho import cli
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


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    """Every test gets its own HOME, so nothing touches the real ~/.claude or ~/Library."""
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


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


# --- the client --------------------------------------------------------------------------------

def no_spawn():
    raise AssertionError("the daemon was started when it should not have been")


def test_listen_prints_only_the_words_and_json_on_request(runtime, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    start(FakeEngine(), idle_minutes=0)
    assert cli.main(["listen"]) == 0
    assert capsys.readouterr().out == "hello there\n"
    assert cli.main(["converse", "Done. Anything else?", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"text": "hello there", "seconds": 1.5}
    ask({"op": "stop"})


def test_errors_map_to_exit_codes_with_nothing_on_stdout(runtime, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    engine = FakeEngine()
    start(engine, idle_minutes=0)
    engine.fail = "timeout"
    assert cli.main(["listen", "--timeout", "1"]) == 2
    assert capsys.readouterr().out == ""
    engine.fail = "mic_denied"
    assert cli.main(["listen"]) == 3
    out, err = capsys.readouterr()
    assert out == "" and "Privacy & Security" in err and "Microphone" in err
    ask({"op": "stop"})


def test_empty_say_fails_without_reaching_the_daemon(runtime, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("   \n"))
    assert cli.main(["say"]) == 1
    assert "nothing to say" in capsys.readouterr().err


def test_stale_socket_is_cleared_and_the_daemon_started(runtime, capsys, monkeypatch):
    protocol.socket_path().write_text("left by a crashed daemon")
    spawned = []

    def spawn():
        spawned.append(True)
        protocol.socket_path().unlink(missing_ok=True)  # serve() also clears it before binding
        start(FakeEngine(), idle_minutes=0)

    monkeypatch.setattr(cli, "_spawn_daemon", spawn)
    assert cli.main(["say", "hello"]) == 0
    assert spawned == [True] and "voice model" in capsys.readouterr().err
    ask({"op": "stop"})


def test_a_daemon_that_dies_mid_request_is_retried_exactly_once(runtime, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(protocol.socket_path()))
    listener.listen()
    accepted = []

    def crash_on_every_request():
        try:
            while len(accepted) < 3:
                conn, _ = listener.accept()
                accepted.append(conn)
                conn.recv(65536)
                conn.close()  # dies before replying
        except OSError:
            pass  # the test closed the listener

    threading.Thread(target=crash_on_every_request, daemon=True).start()
    lock = open(protocol.lock_path(), "w")  # a live daemon holds the lock, so nothing respawns
    __import__("fcntl").flock(lock, __import__("fcntl").LOCK_EX)
    assert cli.main(["say", "hello"]) == 1
    assert len(accepted) == 2
    assert "MacEcho-daemon.log" in capsys.readouterr().err
    lock.close()
    listener.close()


def test_status_and_stop_never_start_a_daemon(runtime, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"running": False}
    assert cli.main(["stop"]) == 0


# --- listening ---------------------------------------------------------------------------------

import wave
from pathlib import Path

import numpy as np

from macecho.daemon import engine
from macecho.vad.vad import VadConfig, VadProcessor

EN_WAV = Path(engine.__file__).parents[1] / "vad" / "en.wav"
RNG = np.random.default_rng(0)


def room(seconds):
    """Quiet room noise: never exact digital zeros, which is what a denied mic delivers."""
    return RNG.normal(0, 20, int(seconds * 16000)).astype(np.int16)


def speech(seconds=4.0):
    with wave.open(str(EN_WAV)) as w:
        return np.frombuffer(w.readframes(int(seconds * 16000)), np.int16)


def frames(*parts):
    audio = np.concatenate(parts)
    return (audio[i:i + 512] for i in range(0, len(audio) - 511, 512))


def vad():
    return VadProcessor(VadConfig(silence_duration=0.6))


def test_capture_returns_the_first_sentence_as_float_audio_to_the_transcriber():
    heard = []

    def transcribe(segment):
        heard.append(segment)
        return "  hello there  "

    assert engine.capture(frames(room(1), speech(), room(2)), vad(), transcribe, timeout=30) == "hello there"
    assert heard[0].dtype == np.float32 and len(heard[0]) > 16000 * 0.5


def test_an_empty_transcript_is_skipped_and_listening_goes_on():
    replies = iter(["", "the real sentence"])
    text = engine.capture(frames(room(1), speech(), room(2), speech(), room(2)), vad(),
                          lambda _: next(replies), timeout=30)
    assert text == "the real sentence"


def test_nobody_speaking_times_out_with_none():
    assert engine.capture(frames(room(5)), vad(), lambda _: "never", timeout=2) is None


def test_a_denied_microphone_is_reported_not_waited_out():
    with pytest.raises(server.EngineError) as denied:
        engine.capture(frames(np.zeros(16000 * 3, np.int16)), vad(), lambda _: "never", timeout=60)
    assert denied.value.kind == "mic_denied"
