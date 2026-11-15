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


class Live:
    """What the daemon is doing right now, in the terms the orb page draws (ui/src/main.tsx).

    The request thread writes it and the live socket's threads read it. Each field is replaced
    whole, never changed in place, so a reader sees the old value or the new one, never a mix.
    """
    BARS = 13  # one per bar the page draws; the same as menubar.ORB_BARS

    def __init__(self):
        self.state = "off"                      # off, listening or replying
        self.heard = np.zeros(0, np.int16)      # the newest mic frame
        self.saying = (np.zeros(0, np.float32), 0.0, 24000)  # chunk being played, when, its rate

    def snapshot(self) -> dict:
        from macecho.device.device import bands, level

        chunk, began, rate = self.saying
        at = int((time.monotonic() - began) * rate)  # a blocking write plays at playback speed
        said = chunk[at:at + rate // 20]             # the 50 ms leaving the speaker now
        return {"state": self.state, "mic": level(self.heard), "speaker": level(said),
                # each bar shows whoever is louder in that band: the user, or the reply
                "bands": list(map(max, bands(self.heard, RATE, self.BARS), bands(said, rate, self.BARS)))}


class Engine:
    """What the server calls: listen(timeout) -> dict, say(text), status() -> dict."""

    def __init__(self, voice_name: str):
        from macecho.daemon import voice
        from macecho.speech_model import DEFAULT_VOICE, SpeechModel
        from macecho.vad.vad import VadConfig, VadProcessor

        self.voice_name = voice_name
        # "system" speaks with macOS; the model still hears, in its default voice's company.
        self.speech = SpeechModel(DEFAULT_VOICE if voice_name == "system" else voice_name)
        self.vad = VadProcessor(VadConfig(sampling_rate=RATE, silence_duration=0.6))
        self.live = Live()
        self.voice = voice.make(voice_name, self.speech, self.live)

    def listen(self, timeout: float) -> dict:
        import sounddevice as sd
        from macecho.device.device import refresh_devices

        refresh_devices()  # pick up headphones plugged in since the last request
        self.vad.reset()
        start = time.monotonic()
        self.live.state = "listening"
        try:
            with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME) as mic:
                def frames():
                    while True:
                        data, _overflowed = mic.read(FRAME)
                        self.live.heard = data[:, 0].copy()
                        yield self.live.heard

                text = capture(frames(), self.vad, self._transcribe, timeout)
        finally:
            self.live.state, self.live.heard = "off", np.zeros(0, np.int16)
        if text is None:
            raise EngineError("timeout", f"nobody spoke within {timeout:g} s")
        return {"text": text, "seconds": round(time.monotonic() - start, 2)}

    def _transcribe(self, segment: np.ndarray) -> str:
        """The model's transcription, logged with what it heard, so a false trigger is traceable."""
        text = self.speech.transcribe(segment)
        rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) else 0.0
        print(f"heard {len(segment) / RATE:.2f}s (rms {rms:.4f}): {text!r}", flush=True)
        return text

    def say(self, text: str) -> None:
        from macecho.device.device import refresh_devices

        refresh_devices()
        self.live.state = "replying"
        try:
            self.voice.speak(text)
        finally:
            self.live.state = "off"

    def status(self) -> dict:
        import sounddevice as sd

        return {"model": "Fluid Speech Pro + ZONOS3-Mini", "voice": self.voice_name, "pid": os.getpid(),
                "input": sd.query_devices(kind="input")["name"],
                "output": sd.query_devices(kind="output")["name"]}
