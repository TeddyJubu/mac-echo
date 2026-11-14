"""MacEcho app: start/stop the LFM3 voice assistant from a small window or the menu bar.

Run with:  PYTHONPATH=src .venv/bin/python -m macecho.menubar   (or build dist/MacEcho.app
with scripts/build_app.sh; that one has a Dock icon you can keep in the Dock). Cocoa owns the
main thread; the agent's asyncio loop runs on a background thread, and a 0.3 s timer mirrors
the agent's state into the window and the menu bar icon.

The window is one web page (ui/, built to src/macecho/ui/index.html by scripts/build_app.sh)
in a web view: the orb, frequency bars, state, the Start/Stop control and the model switch. The
0.3 s timer sends it the app's state, a second 20 Hz timer mic and speaker loudness, and the
page's clicks come back through PageBridge.
"""

import asyncio
import gc
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import objc
import rumps
from AppKit import NSAppearance, NSApplication, NSColor, NSMenu, NSWindow
from Foundation import NSObject
from WebKit import WKWebsiteDataStore, WKWebView, WKWebViewConfiguration

from macecho.config import MacEchoConfig
from macecho.device.device import bands, level

MODELS = {
    "LFM3-Audio-2B": "LiquidAI/LFM3-Audio-2B-MLX-4bit",
    "LFM3-Omni-2B": "LiquidAI/LFM3-Omni-2B-MLX-8bit",
}
LOG = Path.home() / "Library/Logs/MacEcho.log"
PAGE = Path(__file__).with_name("ui") / "index.html"  # built from ui/ by scripts/build_app.sh
WINDOW = (320, 500)  # content size; the page lays itself out for exactly this
INK = (8, 9, 11)  # the page's background (--ink in ui/src/index.css); the titlebar matches it
ORB_BARS = 13  # how many frequency bars; the page draws one per value it is sent
ICONS = {"loading": "🎙…", "off": "🎙", "listening": "🎙●", "replying": "🎙🔊", "error": "🎙⚠️"}


class NSApp(objc.Category(rumps.rumps.NSApp)):
    """Adds what the window needs to rumps' application delegate (self._app is the app's attributes)."""

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _visible):
        self._app["window"].makeKeyAndOrderFront_(None)  # Dock icon click brings a closed window back
        return False


class PageBridge(NSObject, protocols=[objc.protocolNamed("WKScriptMessageHandler")]):
    """Hears the page's clicks: webkit.messageHandlers.macecho.postMessage(...) in ui/src/main.tsx."""

    def userContentController_didReceiveScriptMessage_(self, _controller, message):
        self.app.on_page(str(message.body()))


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
        self._build_window()
        self.window.makeKeyAndOrderFront_(None)
        self._load(self.repo)
        rumps.Timer(self.refresh, 0.3).start()
        # Loudness has to move faster than state does; only worth sending while a session runs.
        rumps.Timer(lambda _: self.running() and self._push_orb(), 0.05).start()

    def _build_window(self):
        """The window: one dark surface, the page filling it below a titlebar painted to match."""
        web_config = WKWebViewConfiguration.new()
        web_config.setWebsiteDataStore_(WKWebsiteDataStore.nonPersistentDataStore())  # keeps nothing on disk
        self.bridge = PageBridge.alloc().init()
        self.bridge.app = self
        web_config.userContentController().addScriptMessageHandler_name_(self.bridge, "macecho")
        self.page = WKWebView.alloc().initWithFrame_configuration_(((0, 0), WINDOW), web_config)
        self.page.setValue_forKey_(False, "drawsBackground")  # no white flash before the page paints
        if PAGE.exists():
            # As a string, not a file URL: the page is one self-contained file, and this way the
            # web view's own sandboxed process needs no access to the checkout (it is in ~/Desktop).
            self.page.loadHTMLString_baseURL_(PAGE.read_text(), None)
        else:
            print(f"No window page: {PAGE} is missing. Build it with scripts/build_app.sh.", flush=True)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), WINDOW), 1 | 2 | 4, 2, False)  # titled, closable, miniaturizable; buffered
        self.window.setTitle_("MacEcho")  # hidden in the titlebar; still names it in Mission Control
        self.window.setTitleVisibility_(1)  # NSWindowTitleHidden
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.setBackgroundColor_(NSColor.colorWithSRGBRed_green_blue_alpha_(*(c / 255 for c in INK), 1))
        self.window.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        self.window.setReleasedWhenClosed_(False)  # closing only hides it; the app keeps running
        self.window.setContentView_(self.page)
        self.window.setInitialFirstResponder_(self.page)  # Return reaches the page without a click
        self.window.center()

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
        """Main-thread timer: surface agent crashes and mirror state into the window and menu."""
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
        self._push_info()
        self._push_orb()

    def _push_info(self):
        """Tell the page what its text and controls show (window.macechoInfo in ui/src/main.tsx)."""
        if not self.window.isVisible():
            return
        info = {
            "error": self.error,
            "running": self.running(),
            "pending": self.start_when_ready,
            "busy": self.loading,  # on_model ignores a choice made mid-load, so the page disables it
            "models": list(MODELS),
            # None selects nothing: a model set in .env that isn't listed
            "model": next((name for name, repo in MODELS.items() if repo == self.repo), None),
        }
        self.page.evaluateJavaScript_completionHandler_(f"window.macechoInfo?.({json.dumps(info)})", None)

    def _push_orb(self):
        """Tell the orb page the state and how loud the mic and the speaker are (0..1 each)."""
        if not self.window.isVisible():
            return  # a closed window draws nothing
        mic = speaker = 0.0
        bars = [0.0] * ORB_BARS
        if self.running():
            heard, said = self.agent.last_input, self.agent.audio_player.now_playing()
            mic, speaker = level(heard), level(said)
            # Each bar shows whoever is louder in that band: you, or the reply.
            bars = map(max, bands(heard, self.agent.audio_recorder.samplerate, ORB_BARS),
                       bands(said, self.agent.audio_player.samplerate, ORB_BARS))
        # window.macecho is defined by ui/src/main.tsx; "?." covers the moment before it loads.
        self.page.evaluateJavaScript_completionHandler_(
            f"window.macecho?.({self.state()!r}, {mic:.3f}, {speaker:.3f}, "
            f"[{', '.join(f'{bar:.2f}' for bar in bars)}])", None)

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

    def on_page(self, message: str):
        """A click in the page: "toggle", or "model:<name>" for one of MODELS. The same as the menu."""
        if message == "toggle":
            self.on_toggle(None)
        elif (name := message.removeprefix("model:")) in self.model_items:
            self.on_model(self.model_items[name])

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
    nsapp = NSApplication.sharedApplication()
    nsapp.setActivationPolicy_(0)  # regular app: Dock icon and key windows, also under python -m
    # rumps builds no main menu, and without one Cmd-Q and Cmd-W do nothing.
    menu, app_menu = NSMenu.new(), NSMenu.new()
    app_menu.addItemWithTitle_action_keyEquivalent_("Close Window", "performClose:", "w")
    app_menu.addItemWithTitle_action_keyEquivalent_("Quit MacEcho", "terminate:", "q")
    menu.addItemWithTitle_action_keyEquivalent_("", None, "").setSubmenu_(app_menu)
    nsapp.setMainMenu_(menu)
    MacEchoApp().run()


if __name__ == "__main__":
    main()
