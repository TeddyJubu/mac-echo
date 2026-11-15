import asyncio
from types import SimpleNamespace as NS

from macecho.brain import MODEL, Brain, Task


class FakeStream:
    """What `client.messages.stream(...)` returns: text deltas, then the whole message."""

    def __init__(self, deltas, content):
        self.deltas, self.content = deltas, content

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for delta in self.deltas:
                yield delta
        return gen()

    async def get_final_message(self):
        return NS(content=self.content)


class FakeClient:
    """Plays back one (deltas, content) pair per turn, and records what each turn was sent."""

    def __init__(self, *turns):
        self.turns, self.calls, self.messages = list(turns), [], self

    def stream(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})  # a copy: history grows
        return FakeStream(*self.turns.pop(0))


def text(t):
    return NS(type="text", text=t)


def tool(task, session, id="tu_1"):
    return NS(type="tool_use", id=id, name="start_task", input={"task": task, "session": session})


def collect(agen):
    async def go():
        return [item async for item in agen]
    return asyncio.run(go())


def test_sentences_come_out_as_soon_as_they_end():
    client = FakeClient((["Tokyo is the cap", "ital. It has ", "14 million people."],
                         [text("Tokyo is the capital. It has 14 million people.")]))
    assert collect(Brain(client).reply("Capital of Japan?")) == [
        "Tokyo is the capital.", "It has 14 million people."]
    sent = client.calls[0]
    assert sent["model"] == MODEL and sent["thinking"] == {"type": "disabled"}
    assert sent["messages"] == [{"role": "user", "content": "Capital of Japan?"}]
    assert [t["name"] for t in sent["tools"]] == ["start_task"]


def test_a_task_starts_before_the_last_words_are_spoken():
    client = FakeClient((["On it."], [text("On it."), tool("Add a dark mode toggle.", "dark-mode")]))
    brain = Brain(client)
    assert collect(brain.reply("Add dark mode")) == [
        Task("tu_1", "Add a dark mode toggle.", "dark-mode"), "On it."]
    answer = brain.history[-1]
    assert answer["role"] == "user" and answer["content"][0]["tool_use_id"] == "tu_1"


def test_two_tasks_in_one_reply_both_start_and_both_are_answered():
    client = FakeClient(([], [tool("Fix the build.", "build", "tu_1"), tool("Bump the version.", "version", "tu_2")]))
    brain = Brain(client)
    assert [t.session for t in collect(brain.reply("Fix the build and bump the version"))] == ["build", "version"]
    assert [r["tool_use_id"] for r in brain.history[-1]["content"]] == ["tu_1", "tu_2"]  # one message


def test_a_report_is_a_new_turn_that_haiku_answers():
    client = FakeClient((["On it."], [text("On it."), tool("Add dark mode.", "dark-mode")]),
                        (["Dark mode is in."], [text("Dark mode is in.")]))
    brain = Brain(client)
    task = [item for item in collect(brain.reply("Add dark mode")) if isinstance(item, Task)][0]
    assert collect(brain.report(task, "Added the toggle to settings.")) == ["Dark mode is in."]
    assert "Added the toggle to settings." in str(client.calls[1]["messages"][-1])


def test_without_a_project_folder_there_is_no_tool():
    client = FakeClient((["Hi."], [text("Hi.")]))
    collect(Brain(client, can_delegate=False).reply("Hi"))
    assert client.calls[0]["tools"] == [] and "no project folder" in client.calls[0]["system"]


def test_a_blank_base_url_in_the_environment_falls_back_to_anthropic(monkeypatch):
    # Shells started by Claude Code export ANTHROPIC_BASE_URL="", which the SDK would take as the server.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    assert str(Brain().client.base_url).startswith("https://api.anthropic.com")
