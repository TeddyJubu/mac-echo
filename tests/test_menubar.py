import asyncio
import json
import threading
import time
import types

import numpy as np
import rumps

from macecho import menubar
from macecho.device.device import AudioPlayer, bands, level


def make_app():
    app = object.__new__(menubar.MacEchoApp)  # skip __init__: no Cocoa, no model
    app.speech, app.agent, app.future = types.SimpleNamespace(_lock=threading.Lock()), object(), None
    app.loading = app.start_when_ready = False
    app.idle_since, app.repo = 0.0, "repo"  # idle for ages
    return app


def test_idle_model_is_freed_only_when_policy_says_so_and_nothing_is_generating(monkeypatch):
    app = make_app()
    monkeypatch.setattr(menubar, "should_unload", lambda idle: False)
    app._idle_tick()
    assert app.speech is not None
    monkeypatch.setattr(menubar, "should_unload", lambda idle: True)
    with app.speech._lock:  # a reply or transcript is still running
        app._idle_tick()
        assert app.speech is not None
    app._idle_tick()
    assert app.speech is None and app.agent is None  # the agent held the model too


def test_policy_frees_the_model_after_30_idle_minutes():
    assert not menubar.should_unload(29 * 60) and menubar.should_unload(30 * 60)


def test_start_while_unloaded_loads_then_starts(monkeypatch):
    app, loaded, started = make_app(), [], []
    app.speech = None
    monkeypatch.setattr(app, "_load", loaded.append)
    monkeypatch.setattr(app, "_start", lambda: started.append(True))
    app.on_toggle(None)
    assert loaded == ["repo"] and not started
    app.speech = object()  # the load finished
    app._idle_tick()
    assert started == [True] and not app.start_when_ready


def test_window_page_gets_controls_state_and_its_clicks_do_what_the_menu_does(monkeypatch):
    app, toggled, chosen, pushed = make_app(), [], [], []
    app.error = None
    app.model_items = {name: rumps.MenuItem(name) for name in menubar.MODELS}
    app.toggle = rumps.MenuItem("Start Listening")
    app._build_window()
    assert app.window.contentView() is app.page and app.window.titlebarAppearsTransparent()
    app.window = types.SimpleNamespace(isVisible=lambda: True)
    app.page = types.SimpleNamespace(evaluateJavaScript_completionHandler_=lambda js, _: pushed.append(js))

    def sent_info():
        call = next(js for js in reversed(pushed) if js.startswith("window.macechoInfo?."))
        return json.loads(call.removeprefix("window.macechoInfo?.(").removesuffix(")"))

    app.refresh(None)
    assert sent_info() == {"error": None, "running": False, "pending": False, "busy": False,
                           "models": list(menubar.MODELS), "model": None}  # "repo" is not listed
    app.repo, app.error = menubar.MODELS["LFM3-Omni-2B"], "Model failed to load: boom"
    app.refresh(None)
    assert sent_info()["model"] == "LFM3-Omni-2B" and sent_info()["error"] == app.error

    # The page's clicks arrive as script messages and call the menu items' handlers.
    monkeypatch.setattr(app, "on_toggle", toggled.append)
    monkeypatch.setattr(app, "on_model", chosen.append)
    message = lambda body: types.SimpleNamespace(body=lambda: body)
    app.bridge.userContentController_didReceiveScriptMessage_(None, message("toggle"))
    app.bridge.userContentController_didReceiveScriptMessage_(None, message("model:LFM3-Audio-2B"))
    app.bridge.userContentController_didReceiveScriptMessage_(None, message("model:not-a-model"))
    assert toggled == [None] and chosen == [app.model_items["LFM3-Audio-2B"]]  # unknown name ignored


def test_level_reads_silence_as_zero_and_speech_partway():
    t = np.arange(1600) / 16000
    tone = lambda amplitude: (amplitude * 32767 * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    assert level(np.zeros(1600, np.int16)) == 0.0 and level(np.zeros(0, np.int16)) == 0.0
    assert level(tone(0.001)) == 0.0  # room noise, about -63 dB
    assert 0.3 < level(tone(0.05)) < level(tone(0.2)) < 1.0  # quiet speech, then loud speech
    assert level(tone(1.0)) == 1.0
    assert level(tone(0.2)) == level(tone(0.2).astype(np.float32) / 32767)  # float PCM reads the same


def test_bands_put_a_tone_in_its_own_band_and_keep_silence_flat():
    def tone(hz, rate=16000):  # 100 ms of a sine at about -17 dB
        return (0.2 * 32767 * np.sin(2 * np.pi * hz * np.arange(rate // 10) / rate)).astype(np.int16)

    assert bands(np.zeros(1600, np.int16), 16000, 13) == [0.0] * 13
    assert bands(np.zeros(0, np.int16), 16000, 13) == [0.0] * 13
    low, high = bands(tone(150), 16000, 13), bands(tone(3000), 16000, 13)
    assert int(np.argmax(low)) < 3 and int(np.argmax(high)) > 9  # bars run low pitch to high
    assert max(low) > 0.8 and low[-1] == 0.0  # the tone's band is tall, a far one stays down
    assert int(np.argmax(bands(tone(3000, 24000), 24000, 13))) > 9  # same pitch at the speaker's rate


def test_player_reports_the_part_of_the_chunk_now_playing():
    player = AudioPlayer(samplerate=24000)
    assert player.now_playing().size == 0  # nothing played yet
    loud_then_silent = np.concatenate([np.full(24000, 8000, np.int16), np.zeros(24000, np.int16)])
    player.playing = (loud_then_silent.tobytes(), time.monotonic())  # a 2 s chunk starts now
    assert level(player.now_playing()) > 0.5 and player.now_playing().size == 1200  # 50 ms
    player.playing = (player.playing[0], time.monotonic() - 1.5)  # 1.5 s in: the silent half
    assert level(player.now_playing()) == 0.0
    player.playing = (player.playing[0], time.monotonic() - 5)  # long finished
    assert player.now_playing().size == 0


def test_orb_gets_state_and_live_levels_only_while_a_session_runs():
    app, pushed = make_app(), []
    app.error, app.window = None, types.SimpleNamespace(isVisible=lambda: True)
    app.page = types.SimpleNamespace(evaluateJavaScript_completionHandler_=lambda js, _: pushed.append(js))
    flat = ", ".join(["0.00"] * menubar.ORB_BARS)
    app._push_orb()
    assert pushed == [f"window.macecho?.('off', 0.000, 0.000, [{flat}])"]

    t = np.arange(1600) / 16000
    voice = (8000 * np.sin(2 * np.pi * 150 * t)).astype(np.int16)  # a low hum into the mic
    app.future = types.SimpleNamespace(done=lambda: False)  # a session is running
    app.agent = types.SimpleNamespace(
        current_llm_task=None, last_input=voice, audio_recorder=types.SimpleNamespace(samplerate=16000),
        audio_player=types.SimpleNamespace(samplerate=24000, now_playing=lambda: np.zeros(0, np.int16)))
    app._push_orb()
    sent = eval(pushed[-1].removeprefix("window.macecho?."))  # the call's arguments, as a tuple
    assert sent[:3] == ("listening", round(level(voice), 3), 0.0)
    assert len(sent[3]) == menubar.ORB_BARS and max(sent[3][:3]) > 0.5 and sent[3][-1] == 0.0

    app.window = types.SimpleNamespace(isVisible=lambda: False)
    app._push_orb()
    assert len(pushed) == 2  # nothing is sent to a closed window


def test_each_session_rereads_the_audio_devices_before_opening_any_stream(monkeypatch):
    # PortAudio lists devices once per process: without a re-read, headphones plugged in after
    # launch never become the output, and replies go to the speakers the jack has muted.
    from macecho import agent as agent_module
    from macecho.device import device

    order = []
    monkeypatch.setattr(device, "refresh_devices", lambda: order.append("refresh"))
    monkeypatch.setattr(device, "list_devices", lambda: None)

    async def play(_):
        order.append("play")

    async def record():
        order.append("record")
        return
        yield

    async def cleanup(_task):
        if _task:
            await _task

    a = object.__new__(agent_module.Agent)
    a.sounddevice_io = (types.SimpleNamespace(start=record), types.SimpleNamespace(play=play))
    a._open_voice_io = lambda: order.append("voice_io")  # None: no echo cancellation this session
    a.audio_player_queue_iterator, a.cleanup = None, cleanup
    asyncio.run(a.start())
    assert order[0] == "refresh" and set(order) == {"refresh", "voice_io", "play", "record"}
