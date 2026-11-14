import numpy as np
import pytest

AV = pytest.importorskip("AVFoundation")
from macecho.device import voice_io  # noqa: E402


def test_mic_audio_becomes_16k_mono_int16_from_the_first_of_five_channels():
    # Voice processing hands the tap 5 channels at 48 kHz; the app wants 16 kHz mono int16.
    layout = AV.AVAudioChannelLayout.alloc().initWithLayoutTag_((147 << 16) | 5)  # DiscreteInOrder
    source = AV.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_interleaved_channelLayout_(
        AV.AVAudioPCMFormatFloat32, 48000, False, layout)
    converter, target = voice_io.mic_converter(source, 16000)

    pcm = b""
    for _ in range(10):  # 1 s in 100 ms tap buffers, like the engine delivers
        buffer = AV.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(source, 4800)
        buffer.setFrameLength_(4800)
        t = (len(pcm) // 2 * 3 + np.arange(4800)) / 48000
        for c in range(5):  # the signal only on channel 0: the others must not leak in
            np.frombuffer(buffer.floatChannelData()[c].as_buffer(4800), np.float32)[:] = (
                0.5 * np.sin(2 * np.pi * 440 * t) if c == 0 else 0.9)
        pcm += voice_io.convert(converter, target, buffer)

    samples = np.frombuffer(pcm, np.int16)
    assert 15000 < samples.size <= 16000  # the converter holds back a few ms of filter delay
    tone = samples[1000:] / 32768.0
    assert abs(np.sqrt(np.mean(tone ** 2)) - 0.5 / np.sqrt(2)) < 0.01  # a 0.5 sine, nothing added
    assert abs(np.mean(tone)) < 0.01  # channels 1-4 were a constant 0.9: none of it got through


def test_speaker_buffer_holds_the_pcm_as_float():
    io = object.__new__(voice_io.VoiceIO)
    io.speaker_format = AV.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(24000, 1)
    pcm = np.array([0, 16384, -32768, 32767], np.int16)
    buffer = io.speaker_buffer(pcm.tobytes())
    assert buffer.frameLength() == 4
    assert np.allclose(np.frombuffer(buffer.floatChannelData()[0].as_buffer(4), np.float32), pcm / 32768.0)
