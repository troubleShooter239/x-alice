"""Модуль для транскрипции речи."""

from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel


class SpeechTranscriberProtocol(Protocol):
    """Протокол для транскрипции речи."""

    def transcribe(self, audio: np.ndarray) -> str:
        """Транскрибирует аудио в текст."""
        ...


class SpeechTranscriber:
    """Класс для транскрипции речи с использованием Whisper."""

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 1,
    ) -> None:
        """Инициализирует транскрибер.

        Args:
            model_name: Название модели Whisper
            device: Устройство для вычислений (cpu/cuda)
            compute_type: Тип вычислений
            language: Язык для распознавания
            beam_size: Размер луча для декодирования
        """
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio: np.ndarray) -> str:
        """Транскрибирует аудио в текст.

        Args:
            audio: Массив аудио-данных

        Returns:
            Распознанный текст
        """
        segments, _ = self.model.transcribe(audio, beam_size=self.beam_size, language=self.language)
        text = "".join([seg.text for seg in segments]).strip()
        return text
