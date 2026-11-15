"""MacEcho app: start/stop the voice assistant from a small window or the menu bar.

Run with:  PYTHONPATH=src .venv/bin/python -m macecho.menubar   (or build dist/MacEcho.app
with scripts/build_app.sh; that one has a Dock icon you can keep in the Dock). Cocoa owns the
main thread; the agent's asyncio loop runs on a background thread, and a 0.3 s timer mirrors
the agent's state into the window and the menu bar icon.

The window is one web page (ui/, built to src/macecho/ui/index.html by scripts/build_app.sh)
in a web view: the orb, frequency bars, state, the Start/Stop control and the model switch. The
0.3 s timer sends it the app's state, a second 20 Hz timer mic and speaker loudness, and the
page's clicks come back through PageBridge.

With --view (ViewerApp) the window only draws what the macecho daemon is doing: Claude Code's
voice mode. It loads no model and opens no mic; the daemon owns both and streams its state.
"""

import asyncio
import fcntl
import gc
import json
import socket
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
from macecho.daemon import protocol
from macecho.device.device import bands, level

# The page's picker and the menu's submenu choose the ZONOS3 voice (both read word for word when
# heard back; see speech_model.py). Switching changes the next sentence's voice; nothing reloads.
MODELS = {
    "Imogen": "imogen",
    "Cillian": "cillian",
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
    left the disk cache). Quitting needs no policy: the process exits and macOS takes it all back.
    """
    return idle_seconds >= 30 * 60  # ponytail: same 30 min as the CLI daemon's default


class MacEchoApp(rumps.App):
    def __init__(self):
        super().__init__("MacEcho", title=ICONS["loading"], quit_button=None)
        self.config = MacEchoConfig()
        # ponytail: voice picked in the menu lasts for this session; MACECHO_VOICE sets the default
        self.repo = self.config.voice
        self.speech = self.agent = self.future = self.error = None
        self.loading = self.start_when_ready = False
        self.idle_since = time.monotonic()  # when the loaded model last stopped being used

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

        self.toggle = rumps.MenuItem("Start Listening", callback=self.on_toggle)
        self.model_items = {name: rumps.MenuItem(name, callback=self.on_model) for name in MODELS}
        self.menu = [
            self.toggle,
            ("Voice", list(self.model_items.values())),
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
        self._draw(self.state(), mic, speaker, bars)

    def _draw(self, state: str, mic: float, speaker: float, bars):
        # window.macecho is defined by ui/src/main.tsx; "?." covers the moment before it loads.
        self.page.evaluateJavaScript_completionHandler_(
            f"window.macecho?.({state!r}, {mic:.3f}, {speaker:.3f}, "
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
                print(f"Loading Fluid Speech Pro and ZONOS3-Mini (voice: {repo})...", flush=True)
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
        self.repo = MODELS[sender.title]
        if self.speech is not None:
            self.speech.voice = self.repo  # the next sentence is in the new voice; nothing reloads

    def on_quit(self, _):
        if self.running():
            self._stop()
        rumps.quit_application()


class ViewerApp(MacEchoApp):
    """Only the window, drawing what the macecho daemon streams on live.sock.

    Claude Code talks through the daemon (`macecho converse`), which holds the only model and the
    mic. Loading a second model here would cost ~2 GB and fight it for the mic, so this reads the
    daemon's state instead. It has no controls: Claude Code decides when to listen and speak.
    """

    def __init__(self):
        rumps.App.__init__(self, "MacEcho", title=ICONS["off"], quit_button=None)
        self.live = None  # the daemon's newest snapshot; None while no daemon is running
        self.menu = [
            rumps.MenuItem("Open Log", callback=lambda _: subprocess.run(["open", str(protocol.log_path())])),
            rumps.MenuItem("Quit MacEcho", callback=lambda _: rumps.quit_application()),
        ]
        self._build_window()
        self.window.makeKeyAndOrderFront_(None)
        threading.Thread(target=self._read_live, daemon=True).start()
        rumps.Timer(self.refresh, 0.3).start()
        rumps.Timer(lambda _: self._push_orb(), 0.05).start()

    def _read_live(self, retry: float = 0.5):
        """Keep self.live current; reconnect whenever a daemon (re)starts. Runs on its own thread."""
        while True:
            try:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.connect(str(protocol.live_path()))
                    for line in sock.makefile():
                        self.live = json.loads(line)
            except (OSError, ValueError):
                pass  # no daemon yet, or it stopped mid-line
            self.live = None
            time.sleep(retry)

    def state(self) -> str:
        live = self.live  # read once: the reader thread may replace it meanwhile
        return live["state"] if live else "off"

    def refresh(self, _):
        self.title = ICONS.get(self.state(), ICONS["off"])
        self._push_info()

    def _push_info(self):
        if self.window.isVisible():
            info = {"error": None, "running": False, "pending": False, "busy": False, "models": [],
                    "model": None, "viewer": True, "connected": self.live is not None}
            self.page.evaluateJavaScript_completionHandler_(f"window.macechoInfo?.({json.dumps(info)})", None)

    def _push_orb(self):
        live = self.live
        if self.window.isVisible():
            if live:
                self._draw(live["state"], live["mic"], live["speaker"], live["bands"])
            else:
                self._draw("off", 0.0, 0.0, [0.0] * ORB_BARS)

    def on_page(self, message: str):
        pass  # no controls in the viewer; Return in the page does nothing


def take_viewer_lock():
    """The open viewer's lock, or None if a viewer already holds it. Keep it until exit."""
    protocol.runtime_dir().mkdir(parents=True, exist_ok=True)
    lock = open(protocol.viewer_lock_path(), "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock
    except BlockingIOError:
        lock.close()
        return None


def main():
    viewer = "--view" in sys.argv
    if viewer and not (lock := take_viewer_lock()):  # noqa: F841 (held for the process's life)
        return  # a window is already open
    nsapp = NSApplication.sharedApplication()
    nsapp.setActivationPolicy_(0)  # regular app: Dock icon and key windows, also under python -m
    # rumps builds no main menu, and without one Cmd-Q and Cmd-W do nothing.
    menu, app_menu = NSMenu.new(), NSMenu.new()
    app_menu.addItemWithTitle_action_keyEquivalent_("Close Window", "performClose:", "w")
    app_menu.addItemWithTitle_action_keyEquivalent_("Quit MacEcho", "terminate:", "q")
    menu.addItemWithTitle_action_keyEquivalent_("", None, "").setSubmenu_(app_menu)
    nsapp.setMainMenu_(menu)
    (ViewerApp if viewer else MacEchoApp)().run()


if __name__ == "__main__":
    main()
