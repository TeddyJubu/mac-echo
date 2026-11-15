"""`python -m macecho.daemon`: what the `macecho` command starts in the background."""
import os
import sys

from macecho.daemon import protocol, server


def main() -> int:
    voice = os.environ.get("MACECHO_VOICE", protocol.DEFAULT_VOICE)
    idle = float(os.environ.get("MACECHO_IDLE_MINUTES", "30"))

    def build():
        from macecho.daemon.engine import Engine
        print(f"macecho daemon {os.getpid()}: loading speech models (voice: {voice})", flush=True)
        engine = Engine(voice)
        print("ready", flush=True)
        return engine

    if not server.serve(build, idle_minutes=idle):
        print("another macecho daemon is already running", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
