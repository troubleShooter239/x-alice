"""Модуль для работы с аудио-записью."""

from typing import Protocol

import numpy as np
import sounddevice as sd


class AudioRecorderProtocol(Protocol):
    """Протокол для записи аудио."""

    def record(self, duration: float) -> np.ndarray:
        """Записывает аудио указанной длительности."""
        ...


class AudioRecorder:
    """Класс для записи аудио с микрофона."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        """Инициализирует рекордер аудио.

        Args:
            sample_rate: Частота дискретизации
            channels: Количество каналов
        """
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration: float) -> np.ndarray:
        """Записывает аудио указанной длительности.

        Args:
            duration: Длительность записи в секундах

        Returns:
            Массив аудио-данных
        """
        print("🎙 Говори команду...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()
        return audio.squeeze()
