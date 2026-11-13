from macecho.speech_model import SpeechModel


def make_model(max_history=4):
    m = object.__new__(SpeechModel)  # skip __init__: no model download needed for history logic
    m.history, m.max_history, m.system_prompt = [], max_history, "sys"
    return m


def test_history_alternates_caps_and_skips_empty():
    m = make_model()
    m._remember("my color is green", "Got it.")
    m._remember("what is it?", "Green.")
    m._remember("ignored", "")              # empty reply: misfire, not stored
    m._remember("", "Hello!")               # empty transcript gets a placeholder
    assert [h["role"] for h in m.history] == ["user", "assistant"] * 2
    assert m.history[0]["content"] == "what is it?"   # oldest turn trimmed at cap
    assert m.history[2]["content"] == "(inaudible)"
    assert m._build_messages()[0] == {"role": "system", "content": "sys"}
    assert m._build_messages()[-1] == ""    # slot for the new audio
