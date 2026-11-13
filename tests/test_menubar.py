import threading
import types

from macecho import menubar


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
