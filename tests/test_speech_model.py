import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from macecho.speech_model import RATE, SpeechModel, frame_budget


class FakeVoiceModel:
    """mlx-audio's ZONOS3: streams 44.1 kHz chunks of the text read aloud."""
    voices = ["imogen", "wyatt"]

    def __init__(self, chunks=3):
        self.chunks, self.asked = chunks, []

    def generate(self, text, voice, stream, streaming_interval, max_tokens):
        self.asked.append((text, voice, stream))
        self.max_tokens = max_tokens
        for _ in range(self.chunks):
            yield NS(audio=np.full(4410, 0.5, np.float32), sample_rate=44100)  # 0.1 s each


def speech(**kw):
    return SpeechModel("imogen", ears=object(), speaker=FakeVoiceModel(**kw), warm=False)


def test_speech_streams_in_the_players_rate_chunk_by_chunk():
    model = speech()
    chunks = list(model.tts("Hello there."))
    assert model.speaker.asked == [("Hello there.", "imogen", True)]
    assert len(chunks) >= 3 and all(c.dtype == np.float32 for c in chunks)
    assert abs(sum(c.size for c in chunks) - 0.3 * RATE) < 0.01 * RATE  # 0.3 s at 24 kHz, nothing lost


def test_stopping_ends_speech_at_the_next_chunk():
    model, stop = speech(chunks=50), threading.Event()
    spoken = 0
    for chunk in model.tts("A long story.", stop):
        spoken += chunk.size
        stop.set()
    assert spoken < 0.2 * RATE


def test_an_unknown_voice_is_refused_with_the_choices():
    with pytest.raises(ValueError, match="imogen, wyatt"):
        SpeechModel("savannah", ears=object(), speaker=FakeVoiceModel(), warm=False)


def test_a_click_too_short_to_be_speech_is_not_transcribed():
    assert speech().transcribe(np.zeros(800, np.float32)) == ""  # 50 ms: the ears are never called


def test_speech_is_capped_by_text_length_so_the_voice_cannot_babble_on():
    # Clean runs took at most 0.09 s of speech per character; babbles ran 0.14-0.23 s per character.
    assert frame_budget("Which branch am I on?") / 15 < 0.14 * len("Which branch am I on?")
    assert frame_budget("Once upon a time, a curious orange cat named Marmalade found a loose floorboard.") / 15 \
        > 0.09 * 80 + 0.5  # room for the slowest clean reading, and then some
    assert frame_budget("Hi.") / 15 >= 1.5  # short words still get time to be said
    model = speech()
    list(model.tts("Hello there."))
    assert model.speaker.max_tokens == frame_budget("Hello there.")
