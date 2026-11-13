"""MacEcho menu bar app: start/stop the LFM3 voice assistant from the macOS menu bar.

Run with:  PYTHONPATH=src .venv/bin/python -m macecho.menubar   (or build dist/MacEcho.app
with scripts/build_app.sh). Cocoa owns the main thread; the agent's asyncio loop runs on a
background thread, and a 0.3 s timer mirrors the agent's state into the menu bar icon.
"""

import asyncio
import gc
import subprocess
import sys
import threading
import time
from pathlib import Path

import rumps

from macecho.config import MacEchoConfig

MODELS = {
    "LFM3-Audio-2B": "LiquidAI/LFM3-Audio-2B-MLX-4bit",
    "LFM3-Omni-2B": "LiquidAI/LFM3-Omni-2B-MLX-8bit",
}
LOG = Path.home() / "Library/Logs/MacEcho.log"
ICONS = {"loading": "🎙…", "off": "🎙", "listening": "🎙●", "replying": "🎙🔊", "error": "🎙⚠️"}


def should_unload(idle_seconds: float) -> bool:
    """Policy: has the loaded model sat unused (mic off) long enough to free its ~1.9 GB?

    Asked every 0.3 s while the model is loaded and nothing is listening. True drops the model;
    the next Start Listening then waits for a reload (about 1.4 s, ~3 s once the weights have
    left the disk cache).
    """
    # TODO(human): choose the policy. Until then the model stays loaded, as it always has.
    return False


class MacEchoApp(rumps.App):
    def __init__(self):
        super().__init__("MacEcho", title=ICONS["loading"], quit_button=None)
        self.config = MacEchoConfig()
        # ponytail: model picked in the menu lasts for this session; .env sets the default
        self.repo = self.config.speech_model or MODELS["LFM3-Audio-2B"]
        self.speech = self.agent = self.future = self.error = None
        self.loading = self.start_when_ready = False
        self.idle_since = time.monotonic()  # when the loaded model last stopped being used

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

        self.toggle = rumps.MenuItem("Start Listening", callback=self.on_toggle)
        self.model_items = {name: rumps.MenuItem(name, callback=self.on_model) for name in MODELS}
        self.menu = [
            self.toggle,
            ("Model", list(self.model_items.values())),
            None,
            rumps.MenuItem("Open Log", callback=lambda _: subprocess.run(["open", str(LOG)])),
            rumps.MenuItem("Quit MacEcho", callback=self.on_quit),
        ]
        self._load(self.repo)
        rumps.Timer(self.refresh, 0.3).start()

    # --- state -------------------------------------------------------------
    def running(self) -> bool:
        return self.future is not None and not self.future.done()

    def state(self) -> str:
        if self.error:
            return "error"
        if self.loading:
            return "loading"
        if not self.running():
            return "off"  # model loaded and idle, or unloaded until the next start
        task = self.agent.current_llm_task
        return "replying" if task is not None and not task.done() else "listening"

    def refresh(self, _):
        """Main-thread timer: surface agent crashes and mirror state into the menu."""
        if self.future is not None and self.future.done() and not self.future.cancelled():
            if self.future.exception():
                self.error = f"Agent stopped: {self.future.exception()}"
            self.future = None
        self._idle_tick()
        state = self.state()
        self.title = ICONS[state]
        self.toggle.title = "Stop Listening" if self.running() else "Start Listening"
        for name, item in self.model_items.items():
            item.state = MODELS[name] == self.repo

    def _idle_tick(self):
        """Start a session that was waiting for the model; free a model nobody is using."""
        if self.speech is None or self.running():
            self.idle_since = time.monotonic()
        elif self.start_when_ready:
            self.start_when_ready = False
            self._start()
        # A reply or transcript still finishing holds the lock; unloading under it frees nothing.
        elif should_unload(time.monotonic() - self.idle_since) and not self.speech._lock.locked():
            print("Unloading idle model", flush=True)
            self._unload()

    # --- actions -----------------------------------------------------------
    def _unload(self):
        """Drop the model; macOS takes the memory back within ~3 s (measured 1.9 -> 0.1 GB)."""
        self.speech = self.agent = None  # the agent holds the model too
        gc.collect()
        if mx := sys.modules.get("mlx.core"):
            mx.clear_cache()  # without this the freed weights stay in MLX's buffer cache

    def _load(self, repo: str):
        """Load the speech model off the main thread so the menu stays responsive."""
        self._unload()  # also frees a model being switched away from before loading the next
        self.error, self.loading = None, True

        def work():
            try:
                from macecho.speech_model import SpeechModel
                print(f"Loading {repo}...", flush=True)
                self.speech = SpeechModel(repo)
                print("Model ready", flush=True)
            except Exception as e:
                self.error, self.start_when_ready = f"Model failed to load: {e}", False
                print(self.error, flush=True)
            finally:
                self.loading = False

        threading.Thread(target=work, daemon=True).start()

    def _start(self):
        from macecho.agent import Agent
        self.error = None
        self.agent = Agent(self.config, speech=self.speech)
        self.future = asyncio.run_coroutine_threadsafe(self.agent.start(), self.loop)

    def _stop(self):
        agent, future = self.agent, self.future

        def cancel_reply():
            if agent.current_llm_task is not None:
                agent.current_llm_task.cancel()  # its stop event halts MLX generation

        self.loop.call_soon_threadsafe(cancel_reply)
        future.cancel()  # agent.start() cleans up mic + speaker in its finally block

    def on_toggle(self, _):
        if self.running():
            self._stop()
        elif self.speech is not None:
            self._start()
        else:
            self.start_when_ready = True  # _idle_tick() starts as soon as the model is in
            if not self.loading:
                self._load(self.repo)

    def on_model(self, sender):
        repo = MODELS[sender.title]
        if repo == self.repo or self.loading:
            return  # same model, or a load is already in flight
        if self.running():
            self._stop()
        self.repo = repo
        if self.speech is not None:
            self._load(repo)  # an unloaded app loads the new choice on the next start

    def on_quit(self, _):
        if self.running():
            self._stop()
        rumps.quit_application()


def main():
    from AppKit import NSApplication
    # Accessory policy: menu bar icon only, no Dock icon (Info.plist LSUIElement doesn't
    # reach us because the .app execs the venv's python).
    NSApplication.sharedApplication().setActivationPolicy_(1)
    MacEchoApp().run()


if __name__ == "__main__":
    main()
