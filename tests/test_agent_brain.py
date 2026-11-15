import asyncio

import numpy as np
import pytest

from macecho import agent as agent_module
from macecho.agent import CANT_THINK, Agent
from macecho.brain import Task
from macecho.delegate import Result


@pytest.fixture(autouse=True)
def no_acknowledgement(monkeypatch):
    """Most tests are about the reply itself; the acknowledgement tests switch it back on."""
    monkeypatch.setattr(agent_module, "acknowledgement", lambda heard: None)


class FakeSpeech:
    def __init__(self, heard="Add dark mode"):
        self.heard, self.said = heard, []

    def transcribe(self, audio):
        return self.heard

    def tts(self, text, stop=None):
        self.said.append(text)
        yield np.zeros(240, np.float32)


class FakeBrain:
    def __init__(self, fail=False, long_report=False, slow=0.0):
        self.fail, self.long_report, self.slow = fail, long_report, slow
        self.asked, self.reports = [], []

    async def reply(self, text):
        self.asked.append(text)
        await asyncio.sleep(self.slow)  # the network round trip
        if self.fail:
            raise RuntimeError("no API key")
        yield Task("tu_1", "Add a dark mode toggle.", "dark-mode")
        yield "On it."

    async def report(self, task, text):
        self.reports.append((task.session, text))
        yield "Dark mode is in."
        if self.long_report:
            await asyncio.sleep(10)
            yield "And more."


class FakeDelegate:
    async def run(self, request):
        return Result(True, "Added it.", "Added the dark mode toggle to settings.")


def make_agent(**brain):
    agent = object.__new__(Agent)  # no audio devices, VAD or model: just the brain path
    agent.speech, agent.brain = FakeSpeech(), FakeBrain(**brain)
    agent.delegates, agent.background = {"dark-mode": FakeDelegate()}, set()
    agent.brain_lock, agent.audio_player_queue = asyncio.Lock(), asyncio.Queue()
    agent.current_llm_task = None
    return agent


def test_a_task_is_acknowledged_now_and_reported_when_claude_finishes():
    async def go():
        agent = make_agent()
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        await asyncio.gather(*agent.background)
        return agent

    agent = asyncio.run(go())
    assert agent.speech.said == ["On it.", "Dark mode is in."]
    assert agent.brain.reports == [("dark-mode", "Added the dark mode toggle to settings.")]
    assert agent.audio_player_queue.qsize() == 2  # one chunk of speech each


def test_talking_over_a_report_cuts_it_short_without_losing_the_task():
    async def go():
        agent = make_agent(long_report=True)
        job = asyncio.create_task(agent._deliver_report(Task("tu_1", "x", "dark-mode"), "Added it."))
        while not agent.speech.said:
            await asyncio.sleep(0.01)
        agent.current_llm_task.cancel()  # what _cancel_ongoing_processing does when the user talks
        await asyncio.wait_for(job, 2)   # the delivery ends quietly instead of raising
        return agent

    agent = asyncio.run(go())
    assert agent.speech.said == ["Dark mode is in."]
    assert agent.brain.reports == [("dark-mode", "Added it.")]  # Haiku has it, for "what did Claude say?"


def test_a_brain_failure_is_said_out_loud():
    async def go():
        agent = make_agent(fail=True)
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent

    assert asyncio.run(go()).speech.said == [CANT_THINK]


def test_an_empty_transcript_costs_nothing():
    async def go():
        agent = make_agent()
        agent.speech.heard = ""
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent

    agent = asyncio.run(go())
    assert agent.brain.asked == [] and agent.speech.said == []


def test_the_acknowledgement_plays_while_haiku_is_still_thinking(monkeypatch):
    monkeypatch.setattr(agent_module, "acknowledgement", lambda heard: "Mm.")

    async def go():
        agent = make_agent(slow=0.3)
        start = asyncio.get_running_loop().time()
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent, asyncio.get_running_loop().time() - start

    agent, took = asyncio.run(go())
    assert agent.speech.said[:2] == ["Mm.", "On it."]  # the ack first, then the reply
    assert took < 0.5  # overlapped with the 0.3 s round trip, not added after it


def test_no_acknowledgement_when_it_says_none(monkeypatch):
    monkeypatch.setattr(agent_module, "acknowledgement", lambda heard: None)

    async def go():
        agent = make_agent()
        await agent._process_with_brain(np.zeros(16000, np.int16), "c1")
        return agent

    assert asyncio.run(go()).speech.said == ["On it."]
