import os
import sounddevice as sd
import numpy as np
import asyncio
import logging
import time
from typing import AsyncIterator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('Macecho.Device')

# Calibration for level() and bands(): which loudness (dB below full scale) the UI shows as silent
# and as full. Depends on the microphone and its gain; raise the floor if room noise moves the orb.
LEVEL_FLOOR_DB = -50.0
LEVEL_CEIL_DB = -12.0
BANDS_HZ = (100.0, 4000.0)  # the range bands() splits up: where speech carries its energy


def refresh_devices():
    """Re-read the system's audio devices and its current default input and output.

    PortAudio lists devices once per process, so without this, headphones plugged in after
    launch never become the output: replies keep going to the MacBook speakers, which the
    headphone jack mutes. Call only while no stream is open; restarting PortAudio under an open
    stream breaks it.
    """
    # ponytail: re-read at session start only; a device change mid-session needs Stop/Start.
    sd._terminate()
    sd._initialize()


def _unit(pcm: np.ndarray) -> np.ndarray:
    """PCM samples as float32 in -1..1, whichever integer or float type they arrived in."""
    samples = pcm.astype(np.float32)
    return samples if pcm.dtype.kind == "f" else samples / np.iinfo(pcm.dtype).max


def _scale(power):
    """Mean-square power (1.0 is full scale) to 0..1, linear in decibels between floor and ceiling.

    Decibels because hearing is logarithmic: on a plain amplitude scale normal speech would sit
    near the bottom of the range.
    """
    db = 10 * np.log10(np.maximum(power, 1e-12))
    return np.clip((db - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB), 0.0, 1.0)


def level(pcm: np.ndarray) -> float:
    """Loudness of a PCM block as 0..1 for display."""
    if pcm.size == 0:
        return 0.0
    return float(_scale(np.mean(_unit(pcm) ** 2)))


def bands(pcm: np.ndarray, samplerate: int, count: int) -> list[float]:
    """Loudness (0..1) of `count` frequency bands of a mono PCM block, low to high, for display.

    The bands are equal steps in pitch across BANDS_HZ. Each is scaled as if the whole signal
    were as loud as that band (its power times `count`), so the floor and ceiling mean the same
    for a bar as they do for level().
    """
    if pcm.size < 2:
        return [0.0] * count
    samples = _unit(pcm)
    window = np.hanning(samples.size)  # tapers the block's edges, which would otherwise smear every band
    # Power per FFT bin, normalised so the bins of a steady signal add up to its mean-square power.
    power = 2 * np.abs(np.fft.rfft(samples * window)) ** 2 / (samples.size * np.sum(window ** 2))
    edges = np.searchsorted(np.fft.rfftfreq(samples.size, 1 / samplerate), np.geomspace(*BANDS_HZ, count + 1))
    running = np.concatenate([[0.0], np.cumsum(power)])
    return _scale((running[edges[1:]] - running[edges[:-1]]) * count).tolist()


def list_devices():
    # 列出设备列表，并返回
    devices = sd.query_devices()

    default_input = sd.query_devices(kind='input')
    print(f"\n默认输入设备: {default_input['name']}")
    print(f"支持的采样率: {default_input['default_samplerate']}")

    # 打印可用设备信息，方便调试
    print("\n可用音频设备:")
    print(sd.query_devices())

    return devices


class AudioRecorder:
    """音频录制器"""

    def __init__(self, device=None, channels=1, samplerate=16000, blocksize=1600, dtype=np.int16):
        """
        初始化音频录制器

        Args:
            device: 输入设备名称或ID，默认为None（使用系统默认设备）
            channels: 通道数，默认为1（单声道）
            samplerate: 采样率，默认为16000Hz
            blocksize: 块大小，默认为1600（100ms的数据量）
            dtype: 数据类型，默认为np.int16
        """
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype

        self.queue = asyncio.Queue()
        self.stream = None
        self.is_running = False
        self.logger = logging.getLogger('Macecho.Device.Recorder')

    def callback(self, indata, frames, time, status):
        """音频回调函数"""

        if status:
            self.logger.warning(f'状态: {status}')
        # 使用 call_soon_threadsafe 确保线程安全
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, bytes(indata))

    async def start(self) -> AsyncIterator[bytes]:
        """
        启动录音并返回音频数据迭代器

        Yields:
            bytes: 原始音频字节数据
        """
        try:
            # 创建原始输入流
            self.stream = sd.RawInputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                callback=self.callback
            )

            # 获取当前事件循环
            self.loop = asyncio.get_running_loop()

            # 使用上下文管理器确保流的正确关闭
            with self.stream:
                self.is_running = True
                self.logger.info(
                    f'开始录音: 设备={self.device}, 采样率={self.samplerate}, '
                    f'块大小={self.blocksize}'
                )

                while self.is_running:
                    try:
                        # 异步等待音频数据
                        audio_data = await self.queue.get()
                        yield audio_data
                    except asyncio.CancelledError:
                        # 异步取消时退出
                        break
                    except Exception as e:
                        self.logger.error(f'录音错误: {e}')
                        break

        except Exception as e:
            self.logger.error(f'创建录音流错误: {e}')
            raise

        finally:
            await self.stop()
            self.logger.info('录音已停止')

    async def stop(self):
        """停止录音"""
        self.is_running = False
        if self.stream:
            self.stream.close()
            self.stream = None

        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_active(self):
        """检查录音器是否处于活动状态"""
        return self.is_running and self.stream is not None


class AudioPlayer:
    """音频播放器"""

    def __init__(self, device=None, channels=1, samplerate=24000, blocksize=2400, dtype=np.int16):
        """
        初始化音频播放器

        Args:
            device (str): 输出设备名称或ID
            channels (int): 通道数，默认为1（单声道）
            samplerate (int): 采样率，默认为24000Hz
            blocksize (int): 块大小，默认为2400（100ms的数据量）
            dtype (np.dtype): 数据类型，默认为np.int16
        """
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype

        self.queue = asyncio.Queue(maxsize=100)  # 限制队列大小防止内存占用过大
        self.done = asyncio.Event()
        self.playing = (b"", 0.0)  # the chunk being written to the speaker, and when that began
        self.stream = None
        self.is_running = False
        self.logger = logging.getLogger('Macecho.Device.Player')

    async def clear(self):
        """清空播放队列"""
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # 重置事件
        self.done.clear()

        self.logger.debug("播放队列已清空")

    async def play(self, audio_iterator: AsyncIterator[bytes]) -> None:
        """
        播放音频数据

        Args:
            audio_iterator (AsyncIterator[bytes]): 音频数据迭代器
        """
        try:
            # 创建音频流
            self.stream = sd.RawOutputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype
            )

            with self.stream:
                self.is_running = True
                self.logger.info(f'开始播放: 设备={self.device}')
                self.stream.start()

                # 主循环：读取数据并播放
                while not self.done.is_set():
                    try:
                        async for data in audio_iterator:
                            if not self.is_running:
                                break

                            if data is None:  # 结束标记
                                continue

                            try:
                                # Blocking write runs off the event loop: inline, it froze the
                                # loop (mic/VAD, task results) for the length of every chunk.
                                self.playing = (data, time.monotonic())
                                await asyncio.to_thread(self.stream.write, data)
                            except Exception as e:
                                self.logger.error(f'写入音频数据错误: {e}')
                                break

                    except Exception as e:
                        self.logger.error(f'播放循环错误: {e}')
                    finally:
                        self.done.set()

        except Exception as e:
            self.logger.error(f'创建音频流错误: {e}')
            raise

        finally:
            await self.stop()
            self.logger.info('播放已停止')

    def now_playing(self) -> np.ndarray:
        """The 50 ms of samples leaving the speaker now; empty once the last chunk has played.

        A chunk can be seconds long, so this picks the part of it that is due at this moment:
        the blocking write hands samples over at playback speed, so time since it began is the
        position. It leads the sound by the output latency, which is too small to see.
        """
        # ponytail: interleaved when channels > 1, and the meters read it as mono (the default);
        # average the channels here if stereo output is ever configured.
        data, began = self.playing
        samples = np.frombuffer(data, dtype=self.dtype)
        at = int((time.monotonic() - began) * self.samplerate) * self.channels
        return samples[at:at + self.samplerate // 20 * self.channels]

    async def stop(self):
        """停止播放"""
        self.is_running = False
        self.done.set()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # 清空队列
        await self.clear()

    @property
    def is_active(self):
        """检查播放器是否处于活动状态"""
        return self.is_running and self.stream is not None
