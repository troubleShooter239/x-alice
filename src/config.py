# from __future__ import annotations

# from typing import Literal

# from pydantic import Field
# from pydantic_settings import BaseSettings, SettingsConfigDict


# class AudioConfig(BaseSettings):
#     sample_rate: int = Field(default=16000, ge=8000, le=48000)
#     channels: int = Field(default=1)


# class STTConfig(BaseSettings):
#     model_name: Literal[
#         "tiny",
#         "tiny.en",
#         "base",
#         "base.en",
#         "small",
#         "small.en",
#         "distil-small.en",
#         "medium",
#         "medium.en",
#         "distil-medium.en",
#         "large-v1",
#         "large-v2",
#         "large-v3",
#         "large",
#         "distil-large-v2",
#         "distil-large-v3",
#         "large-v3-turbo",
#         "turbo",
#     ] = Field(default="tiny")
#     device: Literal["cpu", "cuda", "auto"] = Field(default="cpu")
#     compute_type: Literal["default", "int8", "int16", "float16", "float32"] = Field(default="int8")
#     language: str | None = Field(default="ru")
#     beam_size: int = Field(default=1, ge=1, le=5)


# class WakeWordConfig(BaseSettings):
#     keyword: str = Field(default="алиса")
#     min_audio_duration: float = Field(default=0.5, ge=0.1, le=2.0)
#     energy_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
#     vad_aggressiveness: int = Field(default=2, ge=0, le=3)
#     cooldown_seconds: float = Field(default=2.0, ge=0.0)
#     buffer_duration_seconds: float = Field(default=1.5, ge=0.5, le=5.0)


# class AssistantConfig(BaseSettings):
#     listen_duration: float = Field(default=3.0, ge=1.0, le=10.0)


# class Config(BaseSettings):
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         env_nested_delimiter="__",
#         case_sensitive=False,
#         extra="ignore",
#     )
#     audio: AudioConfig = Field(default_factory=AudioConfig)
#     stt: STTConfig = Field(default_factory=STTConfig)
#     wake_word: WakeWordConfig = Field(default_factory=WakeWordConfig)
#     assistant: AssistantConfig = Field(default_factory=AssistantConfig)


# config = Config()
