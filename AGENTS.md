# Notes for AI agents

## Voice: talk with the user out loud

If the user wants to talk instead of type ("voice mode", "let's talk", "talk to me"), use the
`macecho` command. It listens to the user and prints what they said, and speaks your replies.

```sh
reply=$(macecho converse "I'm listening. What should we work on?")
```

Run `macecho guide` for the full loop, the exit codes and how to phrase spoken replies. Install it
with `uv tool install git+https://github.com/TeddyJubu/mac-echo` if `macecho` is not on PATH.

For working on this repository itself, see `CLAUDE.md`.
