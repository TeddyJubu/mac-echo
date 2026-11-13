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

    def respond(self, audio: np.ndarray) -> tuple[str, np.ndarray, int]:
        """Blocking. audio: mono float32 at 16 kHz. Returns (text, float32 waveform, sample_rate).

        Does not update history; call remember() afterwards (off the latency-critical path)."""
        from mlx_vlm.generate.audio import generate_audio
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self.processor, self.model.config, self._build_messages(),
                                     num_audios=1, use_tts_template=True)
        # LFM3-Audio (joint text+speech loop) defaults to text-only unless told to interleave;
        # LFM3-Omni writes text then voices it, and would choke on this kwarg.
        extra = {"generation_mode": "interleaved"} if getattr(self.model, "joint_audio_text_generation", False) else {}
        with self._lock:
            result = generate_audio(self.model, self.processor, prompt, audio=[audio],
                                    max_tokens=self.max_tokens, temperature=0.0, **extra)
        text = (result.text or "").strip()
        wav = np.asarray(result.audio, dtype=np.float32) if result.audio is not None else np.zeros(0, np.float32)
        return text, wav, result.sample_rate

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
