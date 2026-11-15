# 定义一个agent类

import asyncio
import threading
import time
import os
import wave
from datetime import datetime
from pathlib import Path
from macecho.config import MacEchoConfig
from macecho.device import device
from macecho.device.device import AudioPlayer, AudioRecorder
from macecho.utils.queue import QueueIterator
import numpy as np
import collections

import contextlib
from macecho.brain import Stop, Task
from macecho.delegate import Delegate
from macecho.vad.vad import VadProcessor
from macecho.vad.interface import VadState
from macecho.sentencizer import LLMSentencizer
from macecho.message import (
    MessageType, MessagePriority,
    create_interrupt_message, create_asr_response,
    create_llm_request, create_llm_response,
    create_tts_request, create_tts_response,
    create_status_message, create_error_message
)

CANT_THINK = "I can't reach Claude right now."


def acknowledgement(heard: str) -> str | None:
    """What LFM3 says the moment the user stops, while Haiku's reply is on its way (~1 s), or None
    to stay quiet. `heard` is the user's transcribed words."""
    # TODO(human)
    return None


async def _prefetched(first, items):
    """`items`, whose first item is already being fetched in `first`, as one async iterator."""
    try:
        yield await first
    except StopAsyncIteration:
        return
    async for item in items:
        yield item


class Agent:
    def __init__(self, config: MacEchoConfig, speech=None):
        self.config = config
        self.asr = None
        self.tts = None
        self.vad = None
        self.audio_player = None
        self.audio_recorder = None
        self.voice_io = None  # Apple's echo-cancelling engine, while a session has it open
        self.last_input = np.zeros(0, np.int16)  # newest microphone block, for the UI's level meter

        # Initialize LLM Sentencizer for streaming sentence processing
        self.sentencizer = LLMSentencizer(
            newline_as_separator=True,
            strip_newlines=True
        )

        # Processing state management
        self.current_llm_task = None
        self.current_tts_tasks = []  # List to track multiple TTS tasks
        self.processing_correlation_id = None  # Track current processing session

        # Message queue for internal communication
        self.message_queue = asyncio.Queue()

        # Initialize VAD processor with mapped config
        from macecho.vad.vad import VadConfig
        vad_config = VadConfig(
            threshold=config.vad.threshold,
            sampling_rate=config.audio_recording.sample_rate,
            padding_duration=config.vad.padding_duration,
            min_speech_duration=config.vad.min_speech_duration,
            silence_duration=config.vad.silence_duration,
            per_frame_duration=config.vad.per_frame_duration,
            model_path=config.vad.model_path
        )
        self.vad = VadProcessor(vad_config)

        self.llm = None
        self.speech = speech  # preloaded SpeechModel (menu bar app reuses it across start/stop)
        if speech is not None:
            pass
        elif config.speech_model:
            # End-to-end LFM3 model replaces ASR + LLM + TTS (heavy legacy deps never imported)
            from macecho.speech_model import SpeechModel
            print(f"Loading speech model {config.speech_model}...")
            self.speech = SpeechModel(config.speech_model)
        else:
            self._init_legacy_pipeline(config)

        self.brain = None  # Claude Haiku when MACECHO_BRAIN=haiku; LFM3 then only hears and speaks
        self.delegates = {}  # Claude Code sessions by name, for the brain's start_task
        self.background = set()  # running Claude Code tasks (held, or asyncio may drop them)
        self.brain_lock = asyncio.Lock()  # one brain turn at a time: replies and reports share history
        if config.brain == "haiku":
            if self.speech is None:
                raise ValueError("MACECHO_BRAIN=haiku needs MACECHO_SPEECH_MODEL: LFM3 hears and speaks for it")
            from macecho.brain import Brain
            self.brain = Brain(model=config.brain_model, can_delegate=bool(config.project_dir))

        self._init_audio_io(config)

    def _init_legacy_pipeline(self, config: MacEchoConfig):
        try:
            import funasr  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "No speech model set and the legacy pipeline (funasr) isn't installed. "
                "Add MACECHO_SPEECH_MODEL=LiquidAI/LFM3-Audio-2B-MLX-4bit to .env") from None
        from macecho.asr.sencevoice.model import SenceVoiceASR
        from macecho.llm import prompt
        from macecho.llm.factory import LLMFactory

        # Initialize ASR processor
        try:
            self.asr = SenceVoiceASR(
                language=config.asr.language,
                sample_rate=config.audio_recording.sample_rate,
                model_dir=config.asr.model_name,
                device=config.asr.device,
                auto_initialize=True
            )
            print(f"ASR initialized: {type(self.asr)}")
        except Exception as e:
            print(f"Warning: Failed to initialize ASR: {e}")
            self.asr = None

        # Initialize LLM processor (MLX Qwen)
        try:
            config.llm.system_prompt = prompt.SYSTEM_PROMPT
            # Convert Pydantic model to dict for LLMFactory
            llm_config_dict = config.llm.model_dump()
            self.llm = LLMFactory.create_llm(llm_config_dict)
            print(f"LLM initialized: {type(self.llm)}")
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")
            self.llm = None

        # Initialize TTS processor (CosyVoice)
        try:
            from macecho.tts import CosyVoiceTTS
            
            self.tts = CosyVoiceTTS(
                voice=config.tts.voice_id or "david",
                sample_rate=config.audio_player.sample_rate,
                api_key=getattr(config.tts, 'api_key', None),
                base_url=getattr(config.tts, 'base_url', None),
            )
            print(f"TTS initialized: { config.tts }")
        except Exception as e:
            print(f"Warning: Failed to initialize TTS: {e}")
            self.tts = None

    def _init_audio_io(self, config: MacEchoConfig):
        # Create debug audio output directory if in debug mode
        if config.debug:
            self.debug_audio_dir = Path("debug_audio")
            self.debug_audio_dir.mkdir(exist_ok=True)
            print(
                f"Debug mode: Audio files will be saved to {self.debug_audio_dir}")
        else:
            self.debug_audio_dir = None
        self.audio_recorder = AudioRecorder(
            device=config.audio_recording.device_index,
            channels=config.audio_recording.channels,
            samplerate=config.audio_recording.sample_rate,
            dtype=config.audio_recording.numpy_dtype,
            blocksize=config.audio_recording.chunk_size)

        self.audio_player = AudioPlayer(
            device=config.audio_player.device_index,
            channels=config.audio_player.channels,
            samplerate=config.audio_player.sample_rate,
            dtype=config.audio_player.numpy_dtype,
            blocksize=config.audio_player.chunk_size)
        self.sounddevice_io = (self.audio_recorder, self.audio_player)  # used when voice_io can't start

        self.audio_player_queue = asyncio.Queue()
        self.audio_player_queue_iterator = QueueIterator(
            self.audio_player_queue)

        # 音频缓冲和分帧相关
        self.frame_duration_ms = 32.0  # 32ms帧长
        self.frame_size = int(self.frame_duration_ms *
                              config.audio_recording.sample_rate / 1000)  # 每帧样本数
        self.audio_buffer = collections.deque(
            maxlen=self.frame_size * 10)  # 缓冲区大小设为10帧
        self.running = False

    async def start(self):
        """Start the audio processing agent with proper exception handling"""
        # No stream is open between sessions: pick up headphones plugged in since the last one.
        device.refresh_devices()
        # 打印所有的设备
        device.list_devices()
        self.voice_io = self._open_voice_io()
        self.audio_recorder, self.audio_player = (
            (self.voice_io.recorder, self.voice_io.player) if self.voice_io else self.sounddevice_io)

        self.running = True
        play_task = None

        try:
            # 创建播放任务
            play_task = asyncio.create_task(
                self.audio_player.play(self.audio_player_queue_iterator))

            # 开始音频录制和处理
            async for audio_data in self.audio_recorder.start():
                if not self.running:
                    break

                try:
                    await self.process_audio(audio_data)
                except Exception as e:
                    print(f"Error processing audio data: {e}")
                    # 继续处理下一个音频块，不中断整个流程
                    continue

        except asyncio.CancelledError:
            print("Audio processing was cancelled")
            raise
        except Exception as e:
            print(f"Critical error in audio processing: {e}")
            raise
        finally:
            # 确保清理资源
            await self.cleanup(play_task)

    def _open_voice_io(self):
        """This session's echo-cancelling audio, or None to use sounddevice without it."""
        output = device.sd.query_devices(kind="output")["name"]
        # Apple's voice processing moves the input to the output's own device: with wired
        # headphones that's the headphone jack, so the Mac's mic went unheard. Headphones can't
        # echo anyway. ponytail: matched by name; macOS calls the jack "External Headphones".
        if "headphones" in output.lower():
            print(f"Echo cancellation off for this session: replies go to {output}")
            return None
        try:
            from macecho.device.voice_io import VoiceIO
            io = VoiceIO(self.config.audio_recording.sample_rate, self.config.audio_player.sample_rate)
            print("Echo cancellation on for this session")
            return io
        except Exception as e:  # no PyObjC AVFoundation, or mic and speaker on different hardware
            print(f"Echo cancellation off for this session: {e}")
            return None

    async def process_audio(self, audio_data: bytes):
        """
        处理输入的音频数据：
        1. 将字节数据转换为numpy数组
        2. 将数据添加到缓冲区
        3. 从缓冲区提取完整的帧进行处理
        4. 对每一帧进行VAD等后续处理

        Args:
            audio_data: 输入的音频数据（字节格式）
        """
        try:
            # 1. 将字节数据转换为numpy数组
            audio_array = np.frombuffer(
                audio_data, dtype=self.config.audio_recording.numpy_dtype)
            self.last_input = audio_array

            # 2. 将数据添加到缓冲区
            self.audio_buffer.extend(audio_array)

            # 3. 从缓冲区提取完整的帧进行处理
            while len(self.audio_buffer) >= self.frame_size:
                # 提取一帧数据
                frame = np.array([self.audio_buffer.popleft()
                                 for _ in range(self.frame_size)])

                # 4. 对帧进行后续处理
                await self._process_frame(frame)

        except Exception as e:
            print(f"Error in process_audio: {e}")
            # 清空缓冲区，避免数据堆积
            self.audio_buffer.clear()

    async def _process_frame(self, frame: np.ndarray):
        """
        处理单个音频帧

        Args:
            frame: 音频帧数据（numpy数组）
        """
        try:
            # Use the initialized VAD processor
            if self.vad:
                speech_segment, state = self.vad.process_audio(frame)

                if state == VadState.SPEECH_START:
                    print(f'VAD: Speech start detected at {time.time():.3f}')

                if state == VadState.SPEECH_END and speech_segment is not None:
                    print(
                        f'VAD: Speech end detected at {time.time():.3f}, segment duration: {len(speech_segment)/self.config.audio_recording.sample_rate:.2f}s')

                    # Save audio in debug mode
                    if self.config.debug and self.debug_audio_dir:
                        await self._save_debug_audio(speech_segment)

                    # Here you can process the complete speech segment
                    # For example: send to ASR, then to LLM, then to TTS
                    await self._process_speech_segment(speech_segment)
            else:
                print('Warning: VAD processor not initialized')

        except Exception as e:
            print(f"Error in _process_frame: {e}")

    async def _process_speech_segment(self, speech_segment: np.ndarray):
        """
        Process a complete speech segment through the pipeline

        Args:
            speech_segment: Complete speech segment detected by VAD
        """
        try:
            # Generate new correlation ID for this processing session
            import uuid
            correlation_id = str(uuid.uuid4())

            # Cancel any ongoing processing when new ASR input arrives
            await self._cancel_ongoing_processing()

            # Set current processing session
            self.processing_correlation_id = correlation_id

            print(f"Processing speech segment of {len(speech_segment)} samples "
                  f"({len(speech_segment)/self.config.audio_recording.sample_rate:.2f}s) "
                  f"[{correlation_id[:8]}]")

            if self.speech:
                # Not awaited: the mic loop keeps running so VAD can hear a barge-in, and
                # _cancel_ongoing_processing() cancels this via current_llm_task.
                handle = self._process_with_brain if self.brain else self._process_speech_model
                self.current_llm_task = asyncio.create_task(handle(speech_segment, correlation_id))
                return

            # Step 1: ASR processing
            transcribed_text = await self._process_asr(speech_segment, correlation_id)
            if not transcribed_text.strip():
                print("ASR: No text transcribed, skipping LLM/TTS")
                return

            # Step 2: LLM processing with streaming sentencizer
            await self._process_llm_streaming(transcribed_text, correlation_id)

        except asyncio.CancelledError:
            print(
                f"Speech processing cancelled [{correlation_id[:8] if 'correlation_id' in locals() else 'unknown'}]")
            raise
        except Exception as e:
            print(f"Error processing speech segment: {e}")
            # Send error message
            error_msg = create_error_message(
                "SpeechProcessingError",
                str(e),
                "agent",
                True
            )
            await self.message_queue.put(error_msg)

    async def _cancel_ongoing_processing(self):
        """Cancel any ongoing LLM and TTS processing"""
        print("🛑 Cancelling ongoing processing...")

        # Cancel current LLM task
        if self.current_llm_task and not self.current_llm_task.done():
            print("   Cancelling LLM task...")
            self.current_llm_task.cancel()
            try:
                await self.current_llm_task
            except asyncio.CancelledError:
                pass
            self.current_llm_task = None

        # Cancel all current TTS tasks
        if self.current_tts_tasks:
            print(f"   Cancelling {len(self.current_tts_tasks)} TTS tasks...")
            for tts_task in self.current_tts_tasks:
                if not tts_task.done():
                    tts_task.cancel()

            # Wait for all TTS tasks to be cancelled
            await asyncio.gather(*self.current_tts_tasks, return_exceptions=True)
            self.current_tts_tasks.clear()

        # Clear audio player queue
        print("   Clearing audio player queue...")
        await self._clear_audio_queue()

        # Send interrupt message
        interrupt_msg = create_interrupt_message("new_asr_input", "vad")
        await self.message_queue.put(interrupt_msg)

    async def _clear_audio_queue(self):
        """Clear the audio player queue"""
        try:
            # Clear all pending audio in the queue
            queue_cleared_count = 0
            while not self.audio_player_queue.empty():
                try:
                    self.audio_player_queue.get_nowait()
                    queue_cleared_count += 1
                except asyncio.QueueEmpty:
                    break

            if queue_cleared_count > 0:
                print(
                    f"   Cleared {queue_cleared_count} audio items from queue")

        except Exception as e:
            print(f"Error clearing audio queue: {e}")

    async def _process_speech_model(self, speech_segment: np.ndarray, correlation_id: str):
        """Speech in -> spoken reply out, in one LFM3 call."""
        audio = speech_segment.astype(np.float32)
        if speech_segment.dtype == np.int16:
            audio /= 32768.0
        loop, stop, start = asyncio.get_running_loop(), threading.Event(), time.time()

        def produce():
            """Worker thread: push each speech chunk to the player as soon as it's decoded."""
            gen, played = self.speech.stream(audio, stop), 0
            while True:
                try:
                    chunk = next(gen)
                except StopIteration as done:
                    return done.value or "", played
                if stop.is_set():
                    continue  # barge-in: drop late chunks; the generator exits on its next step
                if not played:
                    print(f"🔊 first audio after {time.time() - start:.2f}s")
                played += chunk.size
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(self.audio_player_queue.put_nowait, pcm)

        try:
            text, played = await asyncio.to_thread(produce)
        except asyncio.CancelledError:
            stop.set()  # stops MLX generation within one step
            raise
        except Exception as e:  # background task: nobody else would report this
            print(f"Speech model error: {e}")
            return
        print(f"🗣️  {text!r} ({played / 24000:.1f}s audio, done in {time.time() - start:.1f}s)")
        # Transcribe the user's turn for memory while the reply is still playing
        await asyncio.to_thread(self.speech.remember, audio, text)

    async def _process_with_brain(self, speech_segment: np.ndarray, correlation_id: str):
        """LFM3 hears, Claude Haiku thinks (and may hand work to Claude Code), LFM3 speaks."""
        audio = speech_segment.astype(np.float32)
        if speech_segment.dtype == np.int16:
            audio /= 32768.0
        start = time.time()
        text = await asyncio.to_thread(self.speech.transcribe, audio)
        print(f"👂 {text!r} after {time.time() - start:.2f}s")
        if not text:
            return
        async with self.brain_lock:
            items = self.brain.reply(text)
            first = asyncio.ensure_future(anext(items))  # Haiku starts now, while the ack plays
            try:
                if ack := acknowledgement(text):
                    await self._speak(ack, start)
                    start = None
                await self._speak_all(_prefetched(first, items), start)
            except Exception as e:  # no key, no network, an API error: say so instead of going quiet
                print(f"Brain error: {e}")
                await self._speak(CANT_THINK)
            finally:
                first.cancel()  # a barge-in during the ack: don't leave the request running

    async def _speak_all(self, items, since: float | None = None):
        """One brain turn aloud: sentences in order; a hand-over starts Claude Code and moves on."""
        async for item in items:
            if isinstance(item, Task):
                self._start_task(item)
            elif isinstance(item, Stop):
                self._stop_task(item.session)
            else:
                print(f"🗣  {item}", flush=True)
                await self._speak(item, since)
                since = None

    async def _speak(self, sentence: str, since: float | None = None):
        """One sentence in LFM3's voice, onto the player as it's generated. Cancelling stops it."""
        loop, stop = asyncio.get_running_loop(), threading.Event()

        def produce():
            nonlocal since
            for chunk in self.speech.tts(sentence, stop):
                if stop.is_set():
                    return
                if since is not None:
                    print(f"🔊 first audio after {time.time() - since:.2f}s")
                    since = None
                pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                loop.call_soon_threadsafe(self.audio_player_queue.put_nowait, pcm)

        try:
            await asyncio.to_thread(produce)
        except asyncio.CancelledError:
            stop.set()  # stops MLX generation within one step
            raise

    def _start_task(self, task: Task):
        """Claude Code starts on it in the background; barge-ins never cancel this."""
        if task.session not in self.delegates:
            self.delegates[task.session] = Delegate(self.config.project_dir)
        job = asyncio.create_task(self._run_task(self.delegates[task.session], task))
        self.background.add(job)
        job.add_done_callback(self.background.discard)
        job.session = task.session

    def _stop_task(self, session: str):
        """The user said no: Claude is killed (Delegate.run's cancel path) and no report follows."""
        for job in [j for j in self.background if getattr(j, "session", None) == session]:
            print(f"🤖 {session}: stopped by the user")
            job.cancel()

    async def _run_task(self, delegate, task: Task):
        print(f"🤖 {task.session}: {task.task}")
        result = await delegate.run(task.task)
        print(f"🤖 {task.session}: {'done' if result.ok else 'failed'} in {result.seconds:.0f}s, ${result.cost_usd:.2f}")
        await self._deliver_report(task, result.text if result.ok else result.spoken)

    async def _deliver_report(self, task: Task, text: str):
        """Haiku tells the user how Claude's task went, once any reply being spoken has finished."""
        # ponytail: also waits out a long reply; add a priority if reports start feeling late.
        async with self.brain_lock:
            speaking = asyncio.create_task(self._speak_all(self.brain.report(task, text)))
            self.current_llm_task = speaking  # the user can talk over it, as over any reply
            with contextlib.suppress(asyncio.CancelledError):
                await speaking

    async def _process_asr(self, speech_segment: np.ndarray, correlation_id: str) -> str:
        """Process ASR for speech segment"""
        if not (self.asr and self.asr.is_ready()):
            print("ASR: Not available or not ready")
            return ""

        try:
            print("ASR: Starting async transcription...")
            # Convert numpy array to bytes for ASR processing
            if speech_segment.dtype == np.float32:
                # Convert float32 to int16 for ASR
                audio_int16 = (speech_segment * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
            else:
                audio_bytes = speech_segment.tobytes()

            # Use async transcribe to avoid blocking the main thread
            transcribed_text = await self.asr.transcribe(audio_bytes)
            print(f"ASR: {transcribed_text}")

            # Send ASR response message
            asr_response = create_asr_response(
                transcribed_text,
                0.95,  # Placeholder confidence
                1.0,   # Placeholder processing time
                correlation_id
            )
            await self.message_queue.put(asr_response)

            return transcribed_text
        except Exception as e:
            print(f"ASR Error: {e}")
            return ""

    async def _process_llm_streaming(self, text: str, correlation_id: str):
        """Process LLM with streaming output and real-time sentence processing"""
        if not self.llm:
            print("LLM: Not available")
            return

        try:
            print(f"LLM: Starting streaming processing for '{text}'")

            # Reset sentencizer for new conversation
            # Use traditional punctuation-based detection since MLX may not output newlines
            self.sentencizer = LLMSentencizer(
                newline_as_separator=True,  # Use punctuation for MLX models
                strip_newlines=True
            )

            # Create LLM streaming task
            llm_task = asyncio.create_task(
                self._stream_llm_response(text, correlation_id)
            )
            self.current_llm_task = llm_task

            await llm_task

        except asyncio.CancelledError:
            print("LLM: Streaming processing cancelled")
            raise
        except Exception as e:
            print(f"LLM Streaming Error: {e}")
        finally:
            self.current_llm_task = None

    async def _stream_llm_response(self, text: str, correlation_id: str):
        """Stream LLM response and process sentences in real-time"""
        try:

            print("LLM: Starting streaming generation...")

            # Use chat_with_context for better conversation management
            stream_generator = self.llm.chat_with_context(
                user_message=text,
                max_tokens=1000,
                temperature=0.7,
                stream=True
            )

            sentence_count = 0

            # Process streaming chunks
            for chunk_dict in stream_generator:
                # Check for cancellation
                if asyncio.current_task().cancelled():
                    print("LLM: Stream cancelled")
                    break

                # Extract content from the chunk dictionary
                if chunk_dict and chunk_dict.get("choices"):
                    delta = chunk_dict["choices"][0].get("delta", {})
                    content = delta.get("content")

                    if content:
                        # Process chunk through sentencizer (using punctuation detection)
                        sentences = self.sentencizer.process_chunk(content)

                        # Send each complete sentence to TTS immediately
                        for sentence in sentences:
                            if sentence.strip():
                                sentence_count += 1
                                print(
                                    f"LLM → TTS[{sentence_count}]: '{sentence[:50]}'")

                                # Create TTS task for this sentence
                                tts_task = asyncio.create_task(
                                    self._process_single_tts(
                                        sentence, correlation_id, sentence_count - 1)
                                )
                                self.current_tts_tasks.append(tts_task)

            # Process any remaining content in sentencizer buffer
            remaining_sentences = self.sentencizer.finish()
            for sentence in remaining_sentences:
                if sentence.strip():
                    sentence_count += 1
                    print(
                        f"LLM → TTS[{sentence_count}] (final): '{sentence[:50]}'")

                    tts_task = asyncio.create_task(
                        self._process_single_tts(
                            sentence, correlation_id, sentence_count - 1)
                    )
                    self.current_tts_tasks.append(tts_task)

            print(
                f"LLM: Streaming completed, {sentence_count} sentences sent to TTS")

        except asyncio.CancelledError:
            print("LLM: Stream processing cancelled")
            raise
        except Exception as e:
            print(f"LLM Stream Error: {e}")
            raise

    async def _process_tts_sentences(self, text: str, correlation_id: str):
        """Process TTS for text, split into sentences"""
        # Simple sentence splitting (can be improved with proper sentence tokenizer)
        sentences = self._split_into_sentences(text)
        print(f"TTS: Processing {len(sentences)} sentences")

        # Create TTS tasks for each sentence
        tts_tasks = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                task = asyncio.create_task(
                    self._process_single_tts(
                        sentence.strip(), correlation_id, i)
                )
                tts_tasks.append(task)

        # Store tasks for potential cancellation
        self.current_tts_tasks = tts_tasks

        try:
            # Wait for all TTS tasks to complete
            await asyncio.gather(*tts_tasks)
        except asyncio.CancelledError:
            print("TTS: All sentence processing cancelled")
            raise
        finally:
            self.current_tts_tasks.clear()

    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitting"""
        # Basic sentence splitting - can be improved with NLP libraries
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def _process_single_tts(self, sentence: str, correlation_id: str, sentence_index: int):
        """Process TTS for a single sentence"""
        try:
            print(f"TTS[{sentence_index}]: Processing '{sentence[:50]}...'")

            if not self.tts:
                # Placeholder TTS processing when TTS is not available
                print(
                    f"TTS[{sentence_index}]: TTS not available, using placeholder")
                await asyncio.sleep(1.0)

                # Generate placeholder audio (sine wave)
                sample_rate = self.config.audio_player.sample_rate
                duration = min(len(sentence) * 0.1, 5.0)
                samples = int(sample_rate * duration)
                t = np.linspace(0, duration, samples)
                frequency = 440 + sentence_index * 100
                audio_data = (
                    0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

                print(
                    f"TTS[{sentence_index}]: Generated {duration:.1f}s placeholder audio")
                await self.audio_player_queue.put(audio_data.tobytes())

                # Send TTS response message
                tts_response = create_tts_response(
                    audio_data.tobytes(), duration, 1.0, correlation_id
                )
                await self.message_queue.put(tts_response)

            else:
                # Real CosyVoice TTS processing
                print(f"TTS[{sentence_index}]: Synthesizing with CosyVoice...")

                start_time = time.time()

                # Use asyncio.to_thread to avoid blocking the event loop
                audio_data = await asyncio.to_thread(self.tts.synthesize, sentence)

                if audio_data:
                    processing_time = time.time() - start_time
                    print(
                        f"TTS[{sentence_index}]: Synthesized in {processing_time:.2f}s, {len(audio_data)} bytes")

                    # Send audio to player queue
                    await self.audio_player_queue.put(audio_data)

                    # Send TTS response message
                    tts_response = create_tts_response(
                        audio_data, processing_time, processing_time, correlation_id
                    )
                    await self.message_queue.put(tts_response)
                else:
                    print(
                        f"TTS[{sentence_index}]: Synthesis failed, no audio generated")

        except asyncio.CancelledError:
            print(f"TTS[{sentence_index}]: Processing cancelled")
            raise
        except Exception as e:
            print(f"TTS[{sentence_index}] Error: {e}")

    async def _save_debug_audio(self, speech_segment: np.ndarray):
        """
        Save speech segment to WAV file in debug mode

        Args:
            speech_segment: Audio data to save
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                :-3]  # Include milliseconds
            filename = f"speech_{timestamp}.wav"
            filepath = self.debug_audio_dir / filename

            # Convert float32 audio to int16 for WAV format
            if speech_segment.dtype == np.float32:
                # Scale from [-1, 1] to [-32768, 32767]
                audio_int16 = (speech_segment * 32767).astype(np.int16)
            elif speech_segment.dtype == np.int16:
                audio_int16 = speech_segment
            else:
                # Convert other types to float32 first, then to int16
                audio_float = speech_segment.astype(np.float32)
                if audio_float.max() > 1.0 or audio_float.min() < -1.0:
                    # Normalize if values are outside [-1, 1] range
                    audio_float = audio_float / np.max(np.abs(audio_float))
                audio_int16 = (audio_float * 32767).astype(np.int16)

            # Save as WAV file
            with wave.open(str(filepath), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.audio_recording.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

            duration = len(speech_segment) / \
                self.config.audio_recording.sample_rate
            print(
                f"Debug: Saved speech segment to {filepath} (duration: {duration:.2f}s, {len(speech_segment)} samples)")

        except Exception as e:
            print(f"Error saving debug audio: {e}")

    async def stop(self):
        """Stop the audio processing agent gracefully"""
        print("Stopping audio processing agent...")
        self.running = False

        try:
            # 停止音频录制
            if self.audio_recorder:
                await self.audio_recorder.stop()

            # 停止音频播放队列
            await self.audio_player_queue.put(None)  # 发送停止信号

            # 清空音频缓冲区
            self.audio_buffer.clear()

            # Reset VAD if initialized
            if self.vad:
                self.vad.reset()

            # Release ASR resources if initialized
            if self.asr:
                self.asr.release()

            # Cancel any ongoing processing tasks
            if self.current_llm_task and not self.current_llm_task.done():
                self.current_llm_task.cancel()
                try:
                    await self.current_llm_task
                except asyncio.CancelledError:
                    pass

            if self.current_tts_tasks:
                for task in self.current_tts_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self.current_tts_tasks, return_exceptions=True)
                self.current_tts_tasks.clear()

            # Clear message queue
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        except Exception as e:
            print(f"Error during agent stop: {e}")

    async def cleanup(self, play_task=None):
        """Clean up resources"""
        print("Cleaning up audio agent resources...")

        try:
            # 设置停止标志
            self.running = False

            # 取消播放任务
            if play_task and not play_task.done():
                play_task.cancel()
                try:
                    await play_task
                except asyncio.CancelledError:
                    print("Play task cancelled successfully")
                except Exception as e:
                    print(f"Error cancelling play task: {e}")

            # 清理音频录制器
            if self.audio_recorder:
                try:
                    await self.audio_recorder.stop()
                except Exception as e:
                    print(f"Error stopping audio recorder: {e}")

            # 清理音频播放器
            if self.audio_player:
                try:
                    await self.audio_player.stop()
                except Exception as e:
                    print(f"Error stopping audio player: {e}")

            # 清空播放队列
            if hasattr(self, 'audio_player_queue'):
                try:
                    # 清空队列中的所有项目
                    while not self.audio_player_queue.empty():
                        try:
                            self.audio_player_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                except Exception as e:
                    print(f"Error clearing audio player queue: {e}")

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            if self.voice_io:
                self.voice_io.close()
                self.voice_io = None
            print("Cleanup completed")
