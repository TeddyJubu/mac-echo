"""The daemon's engine: the speech model, the VAD, the mic and the voice.

`capture()` is the listening logic on its own (frames in, words out) so it can be tested with a
WAV file; `Engine` wires it to the real microphone and model.
"""
import os
import time

import numpy as np

from macecho.daemon.server import EngineError
from macecho.vad.interface import VadState

RATE, FRAME = 16000, 512  # the VAD's sample rate and frame size (32 ms)


def capture(frames, vad, transcribe, timeout: float) -> str | None:
    """Words of the first non-empty utterance in `frames`, or None after `timeout` seconds.

    Time is counted in audio (samples consumed), which a live mic delivers in real time. A
    sentence still going at the deadline is heard out, not cut off.
    """
    consumed, zeros_so_far, state = 0, 0, VadState.IDLE
    for frame in frames:
        consumed += len(frame)
        if zeros_so_far is not None:
            # A denied mic delivers exact digital zeros; any real room has some noise.
            zeros_so_far = None if np.any(frame) else zeros_so_far + len(frame)
            if zeros_so_far is not None and zeros_so_far >= RATE:
                raise EngineError("mic_denied", "the microphone delivers only silence")
        segment, state = vad.process_audio(frame)
        if state == VadState.SPEECH_END and segment is not None:
            text = transcribe(_as_float(segment)).strip()
            if text:
                return text
            vad.reset()  # a cough or a click: keep listening
            state = VadState.IDLE
        if consumed >= timeout * RATE and state == VadState.IDLE:
            return None
    return None


def _as_float(segment) -> np.ndarray:
    segment = np.asarray(segment)
    if segment.dtype == np.int16:
        return segment.astype(np.float32) / 32768.0
    return segment.astype(np.float32)


class Engine:
    """What the server calls: listen(timeout) -> dict, say(text), status() -> dict."""

    def __init__(self, repo: str, voice_name: str):
        from macecho.daemon import voice
        from macecho.speech_model import SpeechModel
        from macecho.vad.vad import VadConfig, VadProcessor

        self.repo, self.voice_name = repo, voice_name
        self.speech = SpeechModel(repo)
        self.vad = VadProcessor(VadConfig(sampling_rate=RATE, silence_duration=0.6))
        self.voice = voice.make(voice_name, self.speech)

    def listen(self, timeout: float) -> dict:
        import sounddevice as sd
        from macecho.device.device import refresh_devices

        refresh_devices()  # pick up headphones plugged in since the last request
        self.vad.reset()
        start = time.monotonic()
        with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME) as mic:
            def frames():
                while True:
                    data, _overflowed = mic.read(FRAME)
                    yield data[:, 0].copy()

            text = capture(frames(), self.vad, self.speech.transcribe, timeout)
        if text is None:
            raise EngineError("timeout", f"nobody spoke within {timeout:g} s")
        return {"text": text, "seconds": round(time.monotonic() - start, 2)}

    def say(self, text: str) -> None:
        from macecho.device.device import refresh_devices

        refresh_devices()
        self.voice.speak(text)

    def status(self) -> dict:
        import sounddevice as sd

        return {"model": self.repo, "voice": self.voice_name, "pid": os.getpid(),
                "input": sd.query_devices(kind="input")["name"],
                "output": sd.query_devices(kind="output")["name"]}
