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
    # The loser of a start-up race must not load the 1.9 GB model before finding out.
    assert server.serve(lambda: pytest.fail("built an engine without the lock"), idle_minutes=0) is False
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


# --- speaking ----------------------------------------------------------------------------------

from macecho.daemon import voice


def test_spoken_text_drops_markdown_code_and_links():
    assert voice.speakable("`npm test` **passed**, see https://example.com/run/42") == "npm test passed, see"
    assert voice.speakable("Here:\n```py\nprint(1)\n```\nDone.") == "Here: Done."
    assert voice.speakable("Read [the docs](https://x.y) first") == "Read the docs first"
    assert voice.speakable("## Summary\n- one\n- two") == "Summary one two"


def test_long_text_is_spoken_sentence_by_sentence():
    assert voice.sentences("First one. Second one? Third!") == ["First one.", "Second one?", "Third!"]


def test_system_voice_says_the_cleaned_text(monkeypatch):
    ran = []
    monkeypatch.setattr(voice.subprocess, "run", lambda args, check: ran.append(args))
    voice.SystemVoice().speak("**Tests** passed.")
    voice.SystemVoice().speak("```\nonly code\n```")  # nothing speakable: say nothing
    assert ran == [["say", "Tests passed."]]


def test_voice_names_pick_the_right_voice():
    assert isinstance(voice.make("system", None), voice.SystemVoice)
    assert isinstance(voice.make("lfm3", object()), voice.LfmVoice)
    with pytest.raises(ValueError):
        voice.make("robot", None)


# --- agent guide and Claude Code skill ---------------------------------------------------------

def skill_file(home):
    return home / ".claude/skills/macecho-voice/SKILL.md"


def test_first_run_installs_the_skill_when_claude_code_is_present(home, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_spawn_daemon", no_spawn)
    (home / ".claude").mkdir()
    cli.main(["guide"])
    text = skill_file(home).read_text()
    assert text.startswith("---\nname: macecho-voice\n") and "macecho converse" in text
    assert "MACECHO_NO_SKILL" in capsys.readouterr().err


def test_the_skill_is_left_alone_without_claude_code_when_present_or_opted_out(home, capsys, monkeypatch):
    cli.main(["guide"])
    assert not (home / ".claude").exists()  # no Claude Code: nothing created
    (home / ".claude").mkdir()
    monkeypatch.setenv("MACECHO_NO_SKILL", "1")
    cli.main(["guide"])
    assert not skill_file(home).exists()
    monkeypatch.delenv("MACECHO_NO_SKILL")
    skill_file(home).parent.mkdir(parents=True)
    skill_file(home).write_text("my own edits")
    cli.main(["guide"])
    assert skill_file(home).read_text() == "my own edits"  # never overwritten unasked


def test_install_skill_force_overwrites_and_guide_prints_the_guide(home, capsys):
    skill_file(home).parent.mkdir(parents=True)
    skill_file(home).write_text("old")
    assert cli.main(["install-skill", "--force"]) == 0
    assert "macecho converse" in skill_file(home).read_text()
    capsys.readouterr()
    assert cli.main(["guide"]) == 0
    assert "Exit codes" in capsys.readouterr().out


# --- live state, for a window drawing the orb while Claude Code talks through the daemon ---

def tone(rate, seconds, amplitude=0.5):
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(int(rate * seconds)) / rate)).astype(np.float32)


def test_live_snapshot_shows_what_is_heard_and_what_is_being_said():
    from macecho.daemon.engine import Live

    live = Live()
    quiet = live.snapshot()
    assert quiet["state"] == "off" and quiet["mic"] == quiet["speaker"] == 0.0
    assert len(quiet["bands"]) == Live.BARS and not any(quiet["bands"])

    live.heard = (tone(16000, 0.032) * 32767).astype(np.int16)  # one loud 32 ms mic frame
    live.saying = (tone(24000, 1.0), time.monotonic(), 24000)   # a 1 s chunk that began now
    loud = live.snapshot()
    assert loud["mic"] > 0.5 and loud["speaker"] > 0.5 and max(loud["bands"]) > 0.5
    json.dumps(loud)  # plain floats: it goes over the socket as JSON

    live.saying = (tone(24000, 1.0), time.monotonic() - 2, 24000)  # that chunk finished a second ago
    assert live.snapshot()["speaker"] == 0.0


def test_live_socket_keeps_reporting_while_a_listen_holds_the_daemon(runtime):
    from macecho.daemon.engine import Live

    started, release = threading.Event(), threading.Event()

    class ListeningEngine(FakeEngine):
        live = Live()

        def listen(self, timeout):
            self.live.state = "listening"
            started.set()
            release.wait(5)  # the user hasn't spoken yet
            self.live.state = "off"
            return super().listen(timeout)

    thread = start(ListeningEngine(), idle_minutes=0)
    asking = threading.Thread(target=ask, args=({"op": "listen", "timeout": 5},), daemon=True)
    asking.start()
    assert started.wait(5)
    with socket.socket(socket.AF_UNIX) as window:
        window.connect(str(protocol.live_path()))
        states = {protocol.recv(window)["state"] for _ in range(3)}  # ~150 ms of lines
        assert states == {"listening"}  # answered while listen() still holds the main thread
        release.set()
        asking.join(5)
        deadline = time.time() + 2
        while protocol.recv(window)["state"] != "off":
            assert time.time() < deadline, "live state never went back to off"
    assert ask({"op": "stop"}) == {"ok": True}
    thread.join(5)
    assert not thread.is_alive() and not protocol.live_path().exists()


def test_an_engine_without_live_state_gets_no_live_socket(runtime):
    thread = start(FakeEngine(), idle_minutes=0)
    assert not protocol.live_path().exists()
    ask({"op": "stop"})
    thread.join(5)
