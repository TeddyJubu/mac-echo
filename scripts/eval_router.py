"""Can LFM3 decide who should handle a spoken request: Claude Code, or itself?

    .venv/bin/python scripts/eval_router.py [--model REPO] [--out FILE]   labelled sentences, scored
    .venv/bin/python scripts/eval_router.py --voice                       you speak, it routes live

CLAUDE: needs the user's project, files or computer (change, run, check or explain them). LFM3 can't
see any of that, so answering itself means inventing it.
CHAT: general knowledge or small talk that the voice model can answer on its own.

Two mistakes, not equally bad: CHAT sent to Claude wastes ~$0.70 and ~15 s; CLAUDE kept as chat
gets a confident made-up answer about the user's code. Needs Metal, so no sandbox.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = [  # (what the user said, who should handle it)
    # plain tasks
    ("Add a dark mode toggle to the settings page.", "CLAUDE"),
    ("Fix the failing test.", "CLAUDE"),
    ("Commit my changes and push them.", "CLAUDE"),
    ("Rename the function load to load_model everywhere.", "CLAUDE"),
    ("Write a README for this project.", "CLAUDE"),
    ("Run the benchmark again.", "CLAUDE"),
    ("Delete the unused imports in the agent file.", "CLAUDE"),
    ("Update the version number to 0.3.", "CLAUDE"),
    # tasks phrased as questions or remarks
    ("Can you make the button blue?", "CLAUDE"),
    ("Could you add a test for the delegate?", "CLAUDE"),
    ("I think the build is broken.", "CLAUDE"),
    ("The orb is lagging behind the voice, can you look into it?", "CLAUDE"),
    # questions only the project can answer
    ("Why are the tests failing?", "CLAUDE"),
    ("What does the delegate module do?", "CLAUDE"),
    ("Which branch am I on?", "CLAUDE"),
    ("How many TODOs are left in the code?", "CLAUDE"),
    ("Remind me what we changed today.", "CLAUDE"),
    ("Is the daemon running?", "CLAUDE"),
    ("What's in my downloads folder?", "CLAUDE"),
    ("Did the last commit get pushed?", "CLAUDE"),
    # general knowledge
    ("What's the capital of Japan?", "CHAT"),
    ("What's seven times eight?", "CHAT"),
    ("Explain what a hash map is.", "CHAT"),
    ("How does async work in Python?", "CHAT"),
    ("What's the difference between a list and a tuple?", "CHAT"),
    ("Who wrote Pride and Prejudice?", "CHAT"),
    ("How do I say thank you in Japanese?", "CHAT"),
    ("What's a good name for a cat?", "CHAT"),
    # small talk
    ("Hi, how are you?", "CHAT"),
    ("Tell me a joke.", "CHAT"),
    ("Thanks, that's all for now.", "CHAT"),
    ("Good morning!", "CHAT"),
    ("You sound great today.", "CHAT"),
    # traps
    ("I talked to Claude yesterday.", "CHAT"),
    ("Do you think tabs or spaces are better?", "CHAT"),
    ("What can you do?", "CHAT"),
    ("Can you tell me a fun fact?", "CHAT"),
    ("Is Python faster than JavaScript?", "CHAT"),
    ("What should I work on next?", "CHAT"),
    ("Make it shorter next time.", "CHAT"),
]

RULES = ("You route requests for a voice assistant on a programmer's Mac. Reply with exactly one word.\n"
         "CLAUDE: the request needs the user's code, files, project or computer: changing, running, "
         "checking or explaining them.\n"
         "CHAT: anything you can answer from general knowledge, or small talk.")
EXAMPLES = ("\n\nExamples:\n"
            "\"Add a login page.\" -> CLAUDE\n"
            "\"Why does the app crash on start?\" -> CLAUDE\n"
            "\"What's the tallest mountain?\" -> CHAT\n"
            "\"Thanks, see you later.\" -> CHAT")
# (system prompt, user message). Unframed, LFM3 tends to answer the sentence ("Tokyo", "56")
# instead of routing it, so "framed" quotes it inside the instruction.
FRAME = 'Request: "{text}"\nWho should handle it? Answer with one word: CLAUDE or CHAT.'
PROMPTS = {"zero-shot": (RULES, "{text}"), "few-shot": (RULES + EXAMPLES, "{text}"),
           "framed": (RULES, FRAME), "framed few-shot": (RULES + EXAMPLES, FRAME)}


def classify(speech, text: str, prompt_pair) -> tuple[str, float]:
    """LFM3's routing for one sentence, in text mode, and the seconds it took."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    system, template = prompt_pair
    prompt = apply_chat_template(speech.processor, speech.model.config,
                                 [{"role": "system", "content": system},
                                  {"role": "user", "content": template.format(text=text)}], num_audios=0)
    start = time.perf_counter()
    with speech._lock:
        out = generate(speech.model, speech.processor, prompt, max_tokens=4, temperature=0.0, verbose=False)
    word = (out.text or "").strip().split()[0].strip(".,:!\"'").upper() if (out.text or "").strip() else ""
    return (word if word in ("CLAUDE", "CHAT") else f"?{word or 'empty'}"), time.perf_counter() - start


def score(speech, name: str, system) -> dict:
    rows, wrong = [], []
    for text, expected in CASES:
        got, seconds = classify(speech, text, system)
        rows.append({"text": text, "expected": expected, "got": got, "seconds": round(seconds, 3)})
        if got != expected:
            wrong.append(f"    {expected:6} -> {got:8} {text}")
    n = len(rows)
    sent_wrongly = sum(r["expected"] == "CHAT" and r["got"] == "CLAUDE" for r in rows)
    kept_wrongly = sum(r["expected"] == "CLAUDE" and r["got"] != "CLAUDE" for r in rows)
    times = sorted(r["seconds"] for r in rows)
    result = {"accuracy": round(sum(r["got"] == r["expected"] for r in rows) / n, 3),
              "chat_sent_to_claude": sent_wrongly, "claude_kept_as_chat": kept_wrongly,
              "unparseable": sum(r["got"].startswith("?") for r in rows),
              "median_s": times[n // 2], "max_s": times[-1], "rows": rows}
    print(f"\n{name}: {result['accuracy']:.0%} right of {n}  |  chat sent to Claude: {sent_wrongly}  |  "
          f"Claude work kept as chat: {kept_wrongly}  |  unparseable: {result['unparseable']}  |  "
          f"median {result['median_s'] * 1000:.0f} ms, max {result['max_s'] * 1000:.0f} ms")
    print("\n".join(wrong) or "    (no mistakes)")
    return result


def voice(engine, system):
    print("\nSpeak a sentence; Ctrl-C to stop. Try the CASES above, then your own.")
    while True:
        try:
            heard = engine.listen(60)["text"]
        except Exception as e:  # timeout: nobody spoke
            print(f"  ({e})")
            continue
        got, seconds = classify(engine.speech, heard, system)
        print(f"  heard: {heard!r}\n  -> {got}  ({seconds * 1000:.0f} ms)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="speech model repo (default: MACECHO_SPEECH_MODEL)")
    parser.add_argument("--voice", action="store_true", help="route what you say into the mic, live")
    parser.add_argument("--prompt", choices=PROMPTS, default="few-shot", help="for --voice (few-shot scored best)")
    parser.add_argument("--out", help="write results as JSON to this file")
    args = parser.parse_args()
    from macecho.config import MacEchoConfig
    repo = args.model or MacEchoConfig().speech_model or "LiquidAI/LFM3-Audio-2B-MLX-4bit"
    print(f"Loading {repo}...", flush=True)

    if args.voice:
        from macecho.daemon.engine import Engine
        try:
            voice(Engine(repo, "system"), PROMPTS[args.prompt])
        except KeyboardInterrupt:
            return

    from macecho.speech_model import SpeechModel
    speech = SpeechModel(repo)
    results = {name: score(speech, name, system) for name, system in PROMPTS.items()}
    if args.out:
        Path(args.out).write_text(json.dumps({"model": repo, **results}, indent=2))


if __name__ == "__main__":
    main()
