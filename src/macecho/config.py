from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import numpy as np


class AudioRecordingConfig(BaseModel):
    """Audio recording and playback configuration"""
    sample_rate: int = Field(
        default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(
        default=1024, description="Audio chunk size for streaming")
    format: str = Field(default="int16", description="Audio format")
    device_index: Optional[int] = Field(
        default=None, description="Audio device index")

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v not in [16000, 48000]:  # 24000 is not supported by pyaudio
            raise ValueError(
                'Sample rate must be one of: 16000, 48000')
        return v

    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v < 1 or v > 2:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v

    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert format string to numpy dtype"""
        format_mapping = {
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }
        if self.format not in format_mapping:
            raise ValueError(f"Unsupported audio format: {self.format}. "
                             f"Supported formats: {list(format_mapping.keys())}")
        return np.dtype(format_mapping[self.format])


class AudioPlayerConfig(BaseModel):
    """Audio recording and playback configuration"""
    sample_rate: int = Field(
        default=24000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(
        default=1024, description="Audio chunk size for streaming")
    format: str = Field(default="int16", description="Audio format")
    device_index: Optional[int] = Field(
        default=None, description="Audio device index")

    @field_validator('sample_rate')
    @classmethod
    def validate_sample_rate(cls, v):
        if v not in [16000, 24000, 48000]:
            raise ValueError(
                'Sample rate must be one of: 16000, 24000, 48000')
        return v

    @field_validator('channels')
    @classmethod
    def validate_channels(cls, v):
        if v < 1 or v > 2:
            raise ValueError('Channels must be 1 (mono) or 2 (stereo)')
        return v

    @property
    def numpy_dtype(self) -> np.dtype:
        """Convert dtype string to numpy dtype"""
        format_mapping = {
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
            "uint8": np.uint8,
            "uint16": np.uint16,
        }
        if self.format not in format_mapping:
            raise ValueError(f"Unsupported audio format: {self.format}. "
                             f"Supported formats: {list(format_mapping.keys())}")
        return np.dtype(format_mapping[self.format])


class VADConfig(BaseModel):
    """Voice Activity Detection configuration"""
    threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                             description="VAD detection threshold")
    padding_duration: float = Field(
        default=0.2, ge=0.0, description="Speech padding duration in seconds")
    min_speech_duration: float = Field(
        default=0.1, ge=0.0, description="Minimum speech duration in seconds")
    silence_duration: float = Field(
        default=0.6, ge=0.5, description="Silence duration threshold in seconds")
    per_frame_duration: float = Field(
        default=0.032, gt=0.0, description="Per frame duration in seconds")
    model_path: str = Field(default="silero_vad.onnx",
                            description="Path to VAD model file")


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(
        default=10485760, description="Maximum log file size in bytes (10MB)")
    backup_count: int = Field(
        default=5, description="Number of backup log files to keep")

    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f'Log level must be one of: {allowed_levels}')
        return v.upper()


class MacEchoConfig(BaseSettings):
    """Main MacEcho configuration"""
    # Component configurations
    audio_recording: AudioRecordingConfig = Field(
        default_factory=AudioRecordingConfig)
    audio_player: AudioPlayerConfig = Field(default_factory=AudioPlayerConfig)
    vad: VADConfig = Field(default_factory=VADConfig)

    # System configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Application settings
    app_name: str = Field(default="MacEcho", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    voice: str = Field(
        default="imogen",
        description="ZONOS3-Mini voice that reads replies aloud (imogen or cillian; others in speech_model.py), or "
                    "'system' in the agent CLI for the macOS voice. Claude Haiku thinks (ANTHROPIC_API_KEY); "
                    "the transcript text goes to Anthropic, the audio stays on the Mac.")
    brain_model: str = Field(default="claude-haiku-5-5", description="Claude model when brain is haiku.")
    project_dir: Optional[str] = Field(
        default=None,
        description="Folder Claude Code works in when the haiku brain hands it a task. Unset: no hand-overs.")

    model_config = SettingsConfigDict(
        env_prefix="MACECHO_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @classmethod
    def from_env(cls) -> "MacEchoConfig":
        """Load configuration from environment variables"""
        return cls(_env_file=".env")


# Default configuration instance
default_config = MacEchoConfig()
