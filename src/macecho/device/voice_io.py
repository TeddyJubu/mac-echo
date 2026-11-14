"""Mic and speaker on one AVAudioEngine with Apple's voice processing, which cancels echo.

Voice processing subtracts what the engine plays from what its mic hears, so the assistant
stops hearing (and interrupting) itself on speakers. Both directions must go through the same
engine for that, which is why this replaces sounddevice for the whole session.

It only starts when the default input and output are the same hardware (MacBook mic and
speakers, a headset): with headphones on the jack and the built-in mic, the engine fails with
-10875. That is also the one setup with no echo, so the caller falls back to sounddevice.
"""
import asyncio
import collections
import logging
import time

import numpy as np

from macecho.device.device import AudioPlayer, AudioRecorder

logger = logging.getLogger('Macecho.Device.VoiceIO')


def mic_converter(source_format, samplerate):
    """Converts the voice-processed mic format (5 channels at 48 kHz here) to mono int16."""
    import AVFoundation as AV
    target = AV.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
        AV.AVAudioPCMFormatInt16, samplerate, 1, True)
    converter = AV.AVAudioConverter.alloc().initFromFormat_toFormat_(source_format, target)
    converter.setChannelMap_([0])  # the channels are copies of one processed signal; keep the first
    return converter, target


def convert(converter, target, buffer) -> bytes:
    """One tap buffer through the converter, as int16 PCM bytes.

    The converter keeps its filter state between calls, so feeding it buffer by buffer
    resamples the stream without clicks at the seams.
    """
    import AVFoundation as AV
    out = AV.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(target, buffer.frameLength())
    given = []

    def feed(_count, _status):
        if given:
            return None, AV.AVAudioConverterInputStatus_NoDataNow
        given.append(True)
        return buffer, AV.AVAudioConverterInputStatus_HaveData

    converter.convertToBuffer_error_withInputFromBlock_(out, None, feed)
    return bytes(out.int16ChannelData()[0].as_buffer(out.frameLength()))


class VoiceIO:
    """The engine shared by `recorder` and `player`. Raises if voice processing can't start."""

    def __init__(self, recorder_rate=16000, player_rate=24000):
        import AVFoundation as AV
        self.engine = AV.AVAudioEngine.alloc().init()
        mic = self.engine.inputNode()
        # The playback graph must exist before voice processing is turned on: in the other order
        # the output fails to initialise (-10875) even on the MacBook's own mic and speakers.
        self.node = AV.AVAudioPlayerNode.alloc().init()
        self.engine.attachNode_(self.node)
        self.speaker_format = AV.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(player_rate, 1)
        self.engine.connect_to_format_(self.node, self.engine.mainMixerNode(), self.speaker_format)

        ok, err = mic.setVoiceProcessingEnabled_error_(True, None)  # turns it on for the output too
        if not ok:
            raise RuntimeError(f"voice processing unavailable: {err}")
        # ponytail: Apple's defaults, ducking other apps' audio and gain control included.

        converter, target = mic_converter(mic.outputFormatForBus_(0), recorder_rate)
        self.on_mic = None  # set by the recorder while it listens
        # ponytail: macOS delivers ~100 ms tap buffers whatever size is asked for; fine for VAD.
        def tap(buffer, _when):  # a void block: returning anything but None aborts the app
            if self.on_mic:
                self.on_mic(convert(converter, target, buffer))

        mic.installTapOnBus_bufferSize_format_block_(0, 4800, mic.outputFormatForBus_(0), tap)

        self.engine.prepare()
        ok, err = self.engine.startAndReturnError_(None)
        if not ok:
            mic.removeTapOnBus_(0)
            raise RuntimeError(f"voice processing engine won't start: {err}")
        self.recorder = VoiceRecorder(self, recorder_rate)
        self.player = VoicePlayer(self, player_rate)

    def speaker_buffer(self, pcm: bytes):
        """int16 PCM as a float32 buffer for the player node."""
        import AVFoundation as AV
        samples = np.frombuffer(pcm, np.int16)
        buffer = AV.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(self.speaker_format, samples.size)
        buffer.setFrameLength_(samples.size)
        np.frombuffer(buffer.floatChannelData()[0].as_buffer(samples.size), np.float32)[:] = samples / 32768.0
        return buffer

    def close(self):
        self.on_mic = None
        self.node.stop()
        self.engine.inputNode().removeTapOnBus_(0)
        self.engine.stop()


class VoiceRecorder(AudioRecorder):
    """AudioRecorder's interface, fed by the engine's mic tap instead of a sounddevice stream."""

    def __init__(self, io: VoiceIO, samplerate):
        super().__init__(samplerate=samplerate)
        self.io = io

    async def start(self):
        loop = asyncio.get_running_loop()
        self.io.on_mic = lambda pcm: loop.call_soon_threadsafe(self.queue.put_nowait, pcm)
        self.is_running = True
        try:
            while self.is_running:
                yield await self.queue.get()
        finally:
            await self.stop()

    async def stop(self):
        self.io.on_mic = None
        await super().stop()


class VoicePlayer(AudioPlayer):
    """AudioPlayer's interface, playing through the engine so its audio is cancelled from the mic."""

    def __init__(self, io: VoiceIO, samplerate):
        super().__init__(samplerate=samplerate)
        self.io = io

    async def play(self, audio_iterator):
        loop = asyncio.get_running_loop()
        # One chunk playing and one queued behind it, so there is no gap between chunks.
        # ponytail: a barge-in lets up to two 320 ms chunks finish; flush the node if that's too long.
        room, scheduled = asyncio.Semaphore(2), collections.deque()

        def finished():  # on an engine thread, once a chunk has been played
            scheduled.popleft()
            self.playing = (scheduled[0], time.monotonic()) if scheduled else (b"", 0.0)
            if not loop.is_closed():
                loop.call_soon_threadsafe(room.release)

        self.is_running = True
        self.io.node.play()
        try:
            async for data in audio_iterator:
                if not self.is_running:
                    break
                if data is None:
                    continue
                await room.acquire()
                if not scheduled:
                    self.playing = (data, time.monotonic())
                scheduled.append(data)
                self.io.node.scheduleBuffer_completionHandler_(self.io.speaker_buffer(data), finished)
        finally:
            self.done.set()
            await self.stop()

    async def stop(self):
        self.is_running = False
        self.done.set()
        await self.clear()
