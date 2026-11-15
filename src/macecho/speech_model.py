"""MacEcho's ears and voice, both on Apple Silicon via MLX.

Ears: Fluid Speech Pro (FluidVoice-Labs/fluid-speech-pro-mlx, 8-bit), a Parakeet TDT 0.6B v3 fine-tune
and the most accurate open speech-to-text model under 1B parameters. Measured here: 90-180 ms for 3-7 s
of speech, against 260-950 ms for LFM3. Loaded with fluid_speech_mlx.py, FluidVoice's own loader.

Voice: Zyphra's ZONOS3-Mini (mlx-community/ZONOS3-Mini, mlx-audio), streamed; first audio 0.13-0.5 s.
It speaks at 44.1 kHz; chunks are resampled to RATE, what the players and the orb expect.

The thinking happens elsewhere (brain.py). Both models together take about 1 GB of GPU memory.
"""
import math
import threading

import numpy as np
import soxr

from macecho.daemon.protocol import DEFAULT_VOICE, EARS, VOICE_MODEL  # noqa: F401  (shared with the CLI)

# Voices, measured 2026-11-16 with scripts/benchmark_speech.py (spoken by ZONOS, heard back by Fluid
# Speech): imogen and cillian came back word for word 10/10; wyatt, rupert and savannah sometimes
# repeated a sentence or babbled on after it; willow misspoke. ponytail: no guard against that babble;
# cap generation by text length if a good voice ever starts doing it.
RATE = 24000  # what AudioPlayer, VoiceIO and the orb are built for
MIN_SPEECH = 0.1  # seconds; anything shorter is a click, not words


FRAMES_PER_SECOND = 15  # ZONOS3's audio rows; its max_tokens counts these


def frame_budget(text: str) -> int:
    """The most audio frames `text` may take. Every clean reading measured took at most 0.09 s per
    character; every babble (a repeated sentence, words after the end) 0.14 s or more. So 0.11 s per
    character, plus a little, cuts a babble short without clipping real speech; 1.5 s minimum."""
    return math.ceil(FRAMES_PER_SECOND * max(1.5, 0.11 * len(text) + 0.3))


class SpeechModel:
    """transcribe(16 kHz float32) -> text, and tts(text) -> 24 kHz float32 chunks as they're made."""

    def __init__(self, voice: str = DEFAULT_VOICE, ears=None, speaker=None, warm: bool = True):
        if ears is None or speaker is None:
            import mlx.core as mx
        if ears is None:
            from macecho.fluid_speech_mlx import load
            ears = load(EARS, bits=8)
            # Loading leaves lazy ops (dtype casts) on this thread's MLX stream, and streams are per
            # thread: the app hears from asyncio's workers, where those ops can't run ("There is no
            # Stream(cpu, 1) in current thread"). Warming up on silence never reaches the decoder.
            mx.eval(ears.parameters())
        if speaker is None:
            from mlx_audio.tts import load
            speaker = load(VOICE_MODEL)
            mx.eval(speaker.parameters())
        if voice not in speaker.voices:
            raise ValueError(f"unknown voice {voice!r}; choose one of: {', '.join(sorted(speaker.voices))}")
        self.ears, self.speaker, self.voice = ears, speaker, voice
        # ponytail: one lock, so hearing waits for speaking; split it if transcribing while talking matters.
        self._lock = threading.Lock()  # one MLX generation at a time
        if warm:  # the first call of each compiles its kernels: ~5 s the user would otherwise wait for
            self.transcribe(np.zeros(16000, np.float32))
            for _ in self.tts("Ready."):
                pass

    def transcribe(self, audio: np.ndarray) -> str:
        """Blocking. What the user said, with punctuation; the language is detected by itself."""
        if audio.size < MIN_SPEECH * 16000:
            return ""
        from macecho.fluid_speech_mlx import log_mel
        with self._lock:
            result = self.ears.generate(log_mel(audio.astype(np.float32), self.ears.preprocessor_config))
        return result[0].text.strip()

    def tts(self, text: str, stop: threading.Event | None = None):
        """Blocking generator: `text` read aloud, as float32 chunks at RATE, each as soon as it's made."""
        resampler = None
        with self._lock:
            for part in self.speaker.generate(text=text, voice=self.voice, stream=True, streaming_interval=0.3,
                                              max_tokens=frame_budget(text)):
                if stop is not None and stop.is_set():
                    return
                if resampler is None:  # streaming resampler: no clicks where chunks meet
                    resampler = soxr.ResampleStream(part.sample_rate, RATE, 1, dtype="float32")
                chunk = resampler.resample_chunk(np.asarray(part.audio, np.float32))
                if chunk.size:
                    yield chunk
            if resampler is not None:
                tail = resampler.resample_chunk(np.zeros(0, np.float32), last=True)
                if tail.size:
                    yield tail
