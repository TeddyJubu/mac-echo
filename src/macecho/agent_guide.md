# Talking with the user through MacEcho

`macecho` gives you ears and a voice on this Mac. You stay the brain: MacEcho turns the user's
speech into text for you and speaks your text aloud. Everything runs locally.

## The loop

1. Greet the user and listen, in one call:
   `reply=$(macecho converse "I'm listening. What should we work on?")`
2. Act on `$reply` exactly as if the user had typed it.
3. Answer and listen again, in one call:
   `reply=$(macecho converse "Done, the tests pass. What next?")`
4. When the user says goodbye or asks you to stop listening:
   `macecho say "Okay, talk soon."` and then `macecho stop`.

`macecho listen` waits for the user without speaking first. `macecho say "..."` speaks without
waiting for an answer. Each command returns once the speaking or listening is finished.

## Exit codes

- `0`: stdout is exactly what the user said (`say` prints nothing).
- `2`: nobody spoke before the timeout (120 s; change it with `--timeout SECS`). Listen again, or
  stop if the conversation seems to be over.
- `3`: the microphone is off for this terminal app. Tell the user, in text, to turn it on in
  System Settings › Privacy & Security › Microphone.
- `1`: anything else; the reason is on stderr.

## Speaking well

- Talk like a person: one to three short sentences per turn, one question at a time.
- Never read out code, file paths, diffs, tables or long lists. Put them in your normal text
  output and say "I've put it on screen."
- Markdown, code blocks and links are removed before speaking, but plain sentences sound best.

## Good to know

- The first command loads the voice model, which takes a few seconds (plus a one-time download of
  about 1–2 GB the first time it is used). After that, commands respond immediately.
- The microphone is on only while `listen` or `converse` is waiting for speech.
- `macecho status` shows the model and the audio devices. `macecho stop` frees about 2 GB of
  memory; the background daemon also stops by itself after 30 unused minutes.
- `--json` on `listen` and `converse` prints `{"text": ..., "seconds": ...}`.
