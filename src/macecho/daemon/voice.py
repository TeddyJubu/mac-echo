"""Turning an agent's text into speech: LFM3's own voice, or the macOS system voice.

Agents write markdown. speakable() keeps the words and drops the symbols, code blocks and links,
which read aloud as noise; code belongs on screen.
"""
import re
import subprocess


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


class LfmVoice:
    """LFM3 reading the text aloud (audio-only mode), streamed to the speaker as it's generated."""

    def __init__(self, speech_model):
        self.speech = speech_model

    def speak(self, text: str) -> None:
        import sounddevice as sd

        parts = sentences(speakable(text))
        if not parts:
            return
        rate = self.speech.model.codec().config.sampling_rate
        # Leaving the block stops the stream, which waits for queued audio to finish playing.
        with sd.OutputStream(samplerate=rate, channels=1, dtype="float32") as out:
            for sentence in parts:
                for chunk in self.speech.tts(sentence):
                    out.write(chunk.reshape(-1, 1))


def make(name: str, speech_model):
    if name == "lfm3":
        return LfmVoice(speech_model)
    if name == "system":
        return SystemVoice()
    raise ValueError(f"MACECHO_VOICE must be lfm3 or system, not {name!r}")
