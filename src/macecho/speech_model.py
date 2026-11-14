"""End-to-end speech model (LFM3-Audio-2B / LFM3-Omni-2B via mlx-vlm).

One call replaces ASR -> LLM -> TTS: 16 kHz mic audio in, reply text + 24 kHz audio out.
"""

import threading

import numpy as np

# The first line is LFM3's trained voice-chat trigger; without it the model transcribes/echoes the user.
SYSTEM_PROMPT = ("Respond with interleaved text and audio.\n\n"
                 "You are a friendly voice assistant. Keep replies short.")
ASR_PROMPT = "Perform ASR."  # LFM3's trained transcription trigger (auto-detects language)
# Read the user message aloud. Checked 2026-11-14 with generation_mode="audio": at most one word off
# per sentence when transcribed back, first audio after a median 0.32 s (docs/superpowers/specs).
TTS_PROMPT = "Perform TTS."


class SpeechModel:
    def __init__(self, repo: str, system_prompt: str = SYSTEM_PROMPT,
                 max_tokens: int = 1024, max_history: int = 6):
        from mlx_vlm import load

        self.repo = repo
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_history = max_history
        self.history: list[dict] = []  # prior turns as text messages
        self._lock = threading.Lock()  # one MLX generation at a time (reply vs. background transcript)
        self.model, self.processor = load(repo)
        self._warm_up()

    def _warm_up(self) -> None:
        """One throwaway reply, so the first real turn doesn't pay for the lazy codec load and
        first-use GPU setup."""
        stop, limit = threading.Event(), self.max_tokens
        self.max_tokens = 64  # bounded even if the model never speaks for silence
        try:
            for _ in self.stream(np.zeros(16000, np.float32), stop):
                stop.set()  # the first speech chunk has exercised every stage
        finally:
            self.max_tokens = limit

    def _build_messages(self) -> list:
        """Messages for the next turn; the user's audio is attached as the final empty prompt."""
        return [{"role": "system", "content": self.system_prompt}, *self.history, ""]

    def _remember(self, user_text: str, reply_text: str) -> None:
        if not reply_text:
            return  # a misfire shouldn't pollute the context
        self.history += [{"role": "user", "content": user_text or "(inaudible)"},
                         {"role": "assistant", "content": reply_text}]
        # ponytail: fixed window; summarise older turns if long chats lose the thread
        self.history = self.history[-self.max_history:]

    def stream(self, audio: np.ndarray, stop: threading.Event | None = None,
               chunk_frames: int = 4, context_frames: int = 24, hold_frames: int = 1):
        """Blocking generator. audio: mono float32 at 16 kHz.

        Yields float32 24 kHz waveform chunks as speech is generated; the generator's return
        value (StopIteration.value) is the reply text. Setting ``stop`` ends generation early.
        Does not update history; call remember() afterwards (off the latency-critical path).

        Each codec frame is 80 ms. Chunks are decoded with ``context_frames`` of look-back and
        the newest ``hold_frames`` held until the next decode (their iSTFT overlap isn't final);
        24/1 measured max |diff| 0.011 vs. decoding the whole reply at once.
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        # LFM3-Audio and LFM3-Omni (a subclass) share one joint text+speech loop; without
        # "interleaved" it starts in text mode and never speaks.
        prompt = apply_chat_template(self.processor, self.model.config, self._build_messages(),
                                     num_audios=1, use_tts_template=True)
        tokens = yield from self._speak(self._inputs(prompt, [audio]), "interleaved", stop,
                                        chunk_frames, context_frames, hold_frames)
        text = self.processor.tokenizer.decode(
            [t for t in tokens if t != self.model.config.audio_out_token_id], skip_special_tokens=True)
        return text.strip()

    def tts(self, text: str, stop: threading.Event | None = None):
        """Blocking generator: `text` read aloud, as float32 24 kHz chunks (no history, no reply)."""
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.processor, self.model.config,
                                     [{"role": "system", "content": TTS_PROMPT},
                                      {"role": "user", "content": text}], num_audios=0)
        yield from self._speak(self._inputs(prompt, None), "audio", stop, end_at_audio_eos=True)

    def _inputs(self, prompt, audio):
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        inputs = prepare_inputs(self.processor, audio=audio, prompts=prompt,
                                add_special_tokens=should_add_special_tokens(self.model.config.model_type,
                                                                             self.processor))
        inputs.pop("attention_mask", None)
        return inputs

    def _speak(self, inputs, mode, stop=None, chunk_frames=4, context_frames=24, hold_frames=1,
               end_at_audio_eos=False):
        """Generate in `mode`, yielding speech as waveform chunks; returns the generated tokens.

        Shared by stream() (a spoken reply) and tts() (reading text aloud). In audio-only mode the
        model goes on after the end-of-speech frame, so tts() stops there.
        """
        import mlx.core as mx
        from mlx_vlm.generate.common import generation_stream

        model, cfg = self.model, self.model.config
        input_ids = inputs.pop("input_ids")
        codec = model.codec()
        samples = codec.config.sampling_rate * 80 // 1000  # samples per codec frame

        tokens, frames, emitted = [], [], 0

        def decode_new() -> np.ndarray:
            """Waveform from frame `emitted` onward, decoding all frames so far (held ones too)."""
            start = max(0, emitted - context_frames)
            wav = np.asarray(codec.decode(mx.stack(frames[start:], axis=-1)), dtype=np.float32)
            return wav[(emitted - start) * samples:]

        with self._lock, mx.stream(generation_stream):
            for token, codes in model.stream_slots(
                    input_ids, generation_mode=mode, max_tokens=self.max_tokens,
                    temperature=0.0, **{**model.audio_sampling, **inputs}):
                if stop is not None and stop.is_set():
                    break
                tokens.append(token)
                if codes is None:
                    continue
                if codes[0].item() == cfg.audio_eos_code:
                    if end_at_audio_eos:
                        break
                    continue
                frames.append(codes)
                ready = len(frames) - hold_frames
                # A smaller first chunk was tried: 2 frames starts ~35 ms sooner, but on a busy
                # machine the speaker ran dry before the second chunk in 4 of 30 replies.
                if ready - emitted >= chunk_frames:
                    yield decode_new()[:(ready - emitted) * samples]
                    emitted = ready
            if frames and emitted < len(frames) and not (stop is not None and stop.is_set()):
                yield decode_new()  # flush the tail, held frames included
        return tokens

    def respond(self, audio: np.ndarray) -> tuple[str, np.ndarray, int]:
        """Blocking, non-streaming convenience: (text, whole float32 waveform, sample_rate)."""
        gen, chunks = self.stream(audio), []
        while True:
            try:
                chunks.append(next(gen))
            except StopIteration as done:
                wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
                return done.value or "", wav, self.model.codec().config.sampling_rate

    def transcribe(self, audio: np.ndarray) -> str:
        """Blocking. What the user said, as text, using the same model in ASR mode."""
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.processor, self.model.config,
                                     [{"role": "system", "content": ASR_PROMPT}, ""], num_audios=1)
        with self._lock:
            result = generate(self.model, self.processor, prompt, audio=[audio],
                              max_tokens=256, temperature=0.0)
        return (result.text or "").strip()

    def remember(self, audio: np.ndarray, reply_text: str) -> None:
        """Blocking. Transcribe the user's turn and add it + the reply to history."""
        self._remember(self.transcribe(audio) if reply_text else "", reply_text)
