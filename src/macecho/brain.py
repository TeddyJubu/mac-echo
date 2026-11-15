"""The Mac app's brain: Claude Haiku in the cloud, between Fluid Speech's ears and ZONOS's voice.

The speech models hear and speak on the Mac (speech_model.py); only the thinking happens here, and
with it the choice of when to hand work to Claude Code, through Haiku's one tool, start_task
(the shape Bomby uses: github.com/bomby-app/bomby, MIT). The user's words go to Anthropic as
text; their audio never leaves the Mac.
"""
import re
from dataclasses import dataclass

MODEL = "claude-haiku-5-5"
SYSTEM = (
    "You are MacEcho, a capable voice assistant on the user's Mac. Answer questions, chat, tell "
    "stories, explain and brainstorm yourself, with judgment of your own. Everything you write is "
    "read aloud, so use short, plain sentences: no markdown, lists or code. The user's words come "
    "from speech recognition and can be misheard (\"my story\" for \"tell me a story\"): read them "
    "for their most likely meaning in the conversation, and only ask when you truly can't tell. ")
DELEGATING = (
    "Claude Code is a coding agent that works in the user's project folder. Hand work to it with "
    "start_task only when the request needs that folder: reading or changing its files or code, "
    "running commands, git, or questions only the project can answer. Never hand over anything "
    "you can do yourself, such as writing a story, and don't offer Claude Code unless it's needed. "
    "Say in a few words that you're handing it over. When the user says no, stop or never mind "
    "about work Claude is doing, call stop_task for that session; never say you stopped something "
    "without calling it. Claude's report arrives later as a message; tell the user the outcome in "
    "a sentence or two.")
ALONE = ("There is no project folder set up, so you can't change or check the user's files; "
         "say so briefly if they ask.")
START_TASK = {
    "name": "start_task",
    "description": ("Hand a task to Claude Code, which works in the user's project with its files, "
                    "shell and git. It runs in the background and can take minutes; its report "
                    "arrives later as a message."),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": (
                "What the user wants, with every detail they gave, as one clear request. Claude "
                "Code can't hear this conversation.")},
            "session": {"type": "string", "description": (
                "A short name for the piece of work, like \"dark-mode\". Reuse it for follow-ups "
                "to that work; pick a new one for unrelated work.")},
        },
        "required": ["task", "session"],
    },
}
STOP_TASK = {
    "name": "stop_task",
    "description": ("Stop Claude Code's work in a session, when the user says no, stop, cancel or "
                    "never mind. Its report then never arrives."),
    "input_schema": {
        "type": "object",
        "properties": {"session": {"type": "string", "description": "The session name given to start_task."}},
        "required": ["session"],
    },
}
STOPPED = "Stopped. Claude Code has been told to stop, and no report will come."
STARTED = "Started. Claude Code is working on it; its report will arrive as a later message."
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Task:
    id: str        # the tool call's id
    task: str
    session: str


@dataclass(frozen=True)
class Stop:
    id: str        # the tool call's id
    session: str


class Brain:
    """One call's conversation with Haiku. Not safe for two turns at once: the caller serialises."""

    def __init__(self, client=None, model: str = MODEL, can_delegate: bool = True):
        if client is None:
            import os

            import anthropic  # the brain extra; reads ANTHROPIC_API_KEY
            # A blank ANTHROPIC_BASE_URL (Claude Code's shells export one) would replace the server,
            # and passing None makes the SDK read that same blank variable again.
            base_url = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
            client = anthropic.AsyncAnthropic(base_url=base_url)
        self.client, self.model = client, model
        self.tools = [START_TASK, STOP_TASK] if can_delegate else []
        self.system = SYSTEM + (DELEGATING if can_delegate else ALONE)
        # ponytail: the whole call's history goes with every turn; trim the oldest turns if calls get long.
        self.history: list[dict] = []

    async def reply(self, user_text: str):
        """Sentences to speak as they arrive, and a Task or Stop for each tool call."""
        self.history.append({"role": "user", "content": user_text})
        async for item in self._turn():
            yield item

    async def report(self, task: Task, text: str):
        """Claude Code finished: a new turn in which Haiku tells the user how it went."""
        self.history.append({"role": "user", "content": f"[Claude Code's report, session {task.session}]\n{text}"})
        async for item in self._turn():
            yield item

    async def _turn(self):
        pending = ""
        async with self.client.messages.stream(
                model=self.model, max_tokens=400, system=self.system, tools=self.tools,
                thinking={"type": "disabled"}, messages=self.history) as stream:
            async for delta in stream.text_stream:
                *done, pending = SENTENCE_END.split(pending + delta)  # the last piece may not have ended
                for sentence in done:
                    if sentence.strip():
                        yield sentence.strip()
            message = await stream.get_final_message()

        content = [{"type": "text", "text": b.text} if b.type == "text" else
                   {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                   for b in message.content if (b.type == "text" and b.text) or b.type == "tool_use"]
        if content:
            self.history.append({"role": "assistant", "content": content})
        tasks = [Stop(b.id, b.input["session"]) if b.name == "stop_task" else
                 Task(b.id, b.input["task"], b.input["session"])
                 for b in message.content if b.type == "tool_use"]
        if tasks:  # the API needs every tool call answered in the very next message
            self.history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": t.id, "content": STOPPED if isinstance(t, Stop) else STARTED}
                for t in tasks]})
        # Tasks before the last words: the app starts Claude first, so talking over "On it." can't lose it.
        for task in tasks:
            yield task
        if pending.strip():
            yield pending.strip()
