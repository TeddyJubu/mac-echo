import re

from macecho.speech_model import TTS_PROMPT


def test_speech_names_one_voice_so_every_sentence_sounds_like_the_same_speaker():
    # "Perform TTS." alone let LFM3 pick a voice per call: replies switched between a man and a woman
    # sentence by sentence (measured: 125 Hz vs 190-214 Hz). The model card's form fixes the speaker.
    assert re.fullmatch(r"Perform TTS in [A-Z][\w ]+ using a (male|female) voice\.", TTS_PROMPT)
