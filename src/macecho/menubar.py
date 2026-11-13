"""MacEcho menu bar app: start/stop the LFM3 voice assistant from the macOS menu bar.

Run with:  PYTHONPATH=src .venv/bin/python -m macecho.menubar   (or build dist/MacEcho.app
with scripts/build_app.sh). Cocoa owns the main thread; the agent's asyncio loop runs on a
background thread, and a 0.3 s timer mirrors the agent's state into the menu bar icon.
"""

import asyncio
import subprocess
import threading
from pathlib import Path

import rumps

from macecho.config import MacEchoConfig

MODELS = {
    "LFM3-Audio-2B": "LiquidAI/LFM3-Audio-2B-MLX-4bit",
    "LFM3-Omni-2B": "LiquidAI/LFM3-Omni-2B-MLX-8bit",
}
LOG = Path.home() / "Library/Logs/MacEcho.log"
ICONS = {"loading": "🎙…", "off": "🎙", "listening": "🎙●", "replying": "🎙🔊", "error": "🎙⚠️"}


class MacEchoApp(rumps.App):
    def __init__(self):
        super().__init__("MacEcho", title=ICONS["loading"], quit_button=None)
        self.config = MacEchoConfig()
        # ponytail: model picked in the menu lasts for this session; .env sets the default
        self.repo = self.config.speech_model or MODELS["LFM3-Audio-2B"]
        self.speech = self.agent = self.future = self.error = None

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
        if self.speech is None:
            return "loading"
        if not self.running():
            return "off"
        task = self.agent.current_llm_task
        return "replying" if task is not None and not task.done() else "listening"

    def refresh(self, _):
        """Main-thread timer: surface agent crashes and mirror state into the menu."""
        if self.future is not None and self.future.done() and not self.future.cancelled():
            if self.future.exception():
                self.error = f"Agent stopped: {self.future.exception()}"
            self.future = None
        state = self.state()
        self.title = ICONS[state]
        self.toggle.title = "Stop Listening" if self.running() else "Start Listening"
        for name, item in self.model_items.items():
            item.state = MODELS[name] == self.repo

    # --- actions -----------------------------------------------------------
    def _load(self, repo: str):
        """Load the speech model off the main thread so the menu stays responsive."""
        self.speech, self.error = None, None

        def work():
            try:
                from macecho.speech_model import SpeechModel
                print(f"Loading {repo}...", flush=True)
                self.speech = SpeechModel(repo)
                print("Model ready", flush=True)
            except Exception as e:
                self.error = f"Model failed to load: {e}"
                print(self.error, flush=True)

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
        elif self.speech is None:
            rumps.notification("MacEcho", "", "Model is still loading, try again in a moment.")
        else:
            self._start()

    def on_model(self, sender):
        repo = MODELS[sender.title]
        if repo == self.repo or self.speech is None:
            return  # same model, or a load is already in flight
        if self.running():
            self._stop()
        self.repo = repo
        self._load(repo)

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
