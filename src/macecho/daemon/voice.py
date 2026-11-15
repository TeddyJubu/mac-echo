"""Turning an agent's text into speech: a ZONOS3-Mini voice, or the macOS system voice.

Agents write markdown. speakable() keeps the words and drops the symbols, code blocks and links,
which read aloud as noise; code belongs on screen.
"""
import re
import subprocess
import time


def speakable(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)           # code blocks
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)         # [label](url) -> label
    text = re.sub(r"https?://\S+", " ", text)                     # bare links
    text = re.sub(r"^\s*(#+|[-*+>]|\d+\.)\s+", " ", text, flags=re.M)  # headings, bullets, quotes
    text = re.sub(r"[`*|~]", "", text)                            # inline code, emphasis, tables
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)                  # "passed ," -> "passed,"
    return " ".join(text.split()).rstrip(" ,;:")


def sentences(text: str) -> list[str]:
    """Split for speaking: each sentence starts playing as soon as it is generated."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


class SystemVoice:
    """The voice chosen in System Settings › Accessibility › Spoken Content."""

    def speak(self, text: str) -> None:
        if text := speakable(text):
            subprocess.run(["say", text], check=True)


class ModelVoice:
    """ZONOS3-Mini reading the text aloud, streamed to the speaker as it's generated."""

    def __init__(self, speech_model, live=None):
        self.speech, self.live = speech_model, live  # live: engine.Live, so a window can draw the reply

    def speak(self, text: str) -> None:
        import sounddevice as sd

        parts = sentences(speakable(text))
        if not parts:
            return
        from macecho.speech_model import RATE as rate
        start, samples, dry = time.monotonic(), 0, 0
        # Leaving the block stops the stream, which waits for queued audio to finish playing.
        # latency: write() returns while this much is still queued, so generating the next chunk
        # (~0.1 s per 0.32 s of speech) happens while the speaker plays. At the default (~10 ms)
        # every chunk ran dry first: about 40 dropouts in 13 s of speech.
        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32", latency=0.5) as out:
            for sentence in parts:
                for chunk in self.speech.tts(sentence):
                    if self.live:
                        # ponytail: queued up to 0.5 s ahead, so the orb can lead the sound by ~0.2 s;
                        # add out.latency to the start time if that ever looks off.
                        self.live.saying = (chunk, time.monotonic(), rate)
                    # write() reports an underflow: the speaker ran dry before this chunk came,
                    # which the user hears as the voice dropping out.
                    dry += bool(out.write(chunk.reshape(-1, 1)))
                    samples += chunk.size
        print(f"spoke {samples / rate:.1f}s in {time.monotonic() - start:.1f}s, "
              f"{len(parts)} sentences, {dry} dropouts", flush=True)


def make(name: str, speech_model, live=None):
    """"system": the macOS voice; any other name is a ZONOS3-Mini voice (SpeechModel checks it)."""
    return SystemVoice() if name == "system" else ModelVoice(speech_model, live)
