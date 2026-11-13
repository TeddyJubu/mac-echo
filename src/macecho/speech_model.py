"""End-to-end speech model (LFM3-Audio-2B / LFM3-Omni-2B via mlx-vlm).

One call replaces ASR -> LLM -> TTS: 16 kHz mic audio in, reply text + 24 kHz audio out.
"""

import threading

import numpy as np

# The first line is LFM3's trained voice-chat trigger; without it the model transcribes/echoes the user.
SYSTEM_PROMPT = ("Respond with interleaved text and audio.\n\n"
                 "You are a friendly voice assistant. Keep replies short.")
ASR_PROMPT = "Perform ASR."  # LFM3's trained transcription trigger (auto-detects language)


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
        import mlx.core as mx
        from mlx_vlm.generate.common import generation_stream
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        # LFM3-Audio and LFM3-Omni (a subclass) share one joint text+speech loop; without
        # "interleaved" it starts in text mode and never speaks.
        model, cfg = self.model, self.model.config
        prompt = apply_chat_template(self.processor, cfg, self._build_messages(),
                                     num_audios=1, use_tts_template=True)
        inputs = prepare_inputs(self.processor, audio=[audio], prompts=prompt,
                                add_special_tokens=should_add_special_tokens(cfg.model_type, self.processor))
        input_ids = inputs.pop("input_ids")
        inputs.pop("attention_mask", None)
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
                    input_ids, generation_mode="interleaved", max_tokens=self.max_tokens,
                    temperature=0.0, **{**model.audio_sampling, **inputs}):
                if stop is not None and stop.is_set():
                    break
                tokens.append(token)
                if codes is None or codes[0].item() == cfg.audio_eos_code:
                    continue
                frames.append(codes)
                ready = len(frames) - hold_frames
                if ready - emitted >= chunk_frames:
                    yield decode_new()[:(ready - emitted) * samples]
                    emitted = ready
            if frames and emitted < len(frames) and not (stop is not None and stop.is_set()):
                yield decode_new()  # flush the tail, held frames included
        text = self.processor.tokenizer.decode(
            [t for t in tokens if t != cfg.audio_out_token_id], skip_special_tokens=True)
        return text.strip()

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
