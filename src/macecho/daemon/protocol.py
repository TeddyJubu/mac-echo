"""What the macecho client and daemon share: where the socket lives, and the wire format.

One JSON object per line in each direction, one request per connection. Standard library only,
so the client imports this without loading MLX, numpy or sounddevice.
"""
import json
import os
from pathlib import Path

# The models the daemon loads (macecho.speech_model): Fluid Speech Pro hears, ZONOS3-Mini speaks.
EARS = "FluidVoice-Labs/fluid-speech-pro-mlx"
VOICE_MODEL = "mlx-community/ZONOS3-Mini"
DEFAULT_VOICE = "imogen"
# Error kinds the client maps to their own exit codes; any other kind exits 1.
ERRORS = {"timeout": 2, "mic_denied": 3}


def runtime_dir() -> Path:
    """Holds the socket and the lock. MACECHO_RUNTIME_DIR moves it (tests use a short temp dir)."""
    return Path(os.environ.get("MACECHO_RUNTIME_DIR")
                or Path.home() / "Library/Application Support/MacEcho")


def socket_path() -> Path:
    return runtime_dir() / "daemon.sock"


def live_path() -> Path:
    """Where a window reads what the daemon is doing: one JSON line about every 50 ms."""
    return runtime_dir() / "live.sock"


def lock_path() -> Path:
    return runtime_dir() / "daemon.lock"


def viewer_lock_path() -> Path:
    """Held by the open window (menubar --view), so `macecho show` never opens a second one."""
    return runtime_dir() / "viewer.lock"


def app_log_path() -> Path:
    return Path.home() / "Library/Logs/MacEcho.log"


def log_path() -> Path:
    return Path.home() / "Library/Logs/MacEcho-daemon.log"


def send(sock, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode())


def recv(sock) -> dict | None:
    """The next line as a dict; None if the peer closed first. Raises ValueError on bad JSON."""
    buf = bytearray()
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode())  # UnicodeDecodeError is a ValueError too
