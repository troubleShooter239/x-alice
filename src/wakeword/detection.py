"""Модуль для детекции wake word."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import numpy as np
import sounddevice as sd
import webrtcvad


if TYPE_CHECKING:
    from stt.transcription import SpeechTranscriberProtocol


class WakeWordDetectorProtocol(Protocol):
    """Протокол для детекции wake word."""

    def is_wake_word_detected(self, audio: np.ndarray) -> bool:
        """Проверяет, обнаружен ли wake word в аудио."""
        ...


class WakeWordListenerProtocol(Protocol):
    """Протокол для слушателя wake word."""

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Устанавливает callback для обработки обнаруженного wake word."""
        ...

    def listen(self) -> None:
        """Начинает прослушивание wake word."""
        ...


class KeywordWakeWordDetector:
    """Детектор wake word на основе распознавания ключевого слова."""

    def __init__(
        self,
        transcriber: "SpeechTranscriberProtocol",
        keyword: str = "алиса",
        sample_rate: int = 16000,
        min_audio_duration: float = 0.5,
        energy_threshold: float = 0.01,
        vad_aggressiveness: int = 2,
    ) -> None:
        """Инициализирует детектор wake word на основе ключевого слова.

        Args:
            transcriber: Транскрибер для распознавания речи
            keyword: Ключевое слово для детекции (по умолчанию "алиса")
            sample_rate: Частота дискретизации
            min_audio_duration: Минимальная длительность аудио для распознавания (секунды)
            energy_threshold: Порог энергии для предварительной фильтрации
            vad_aggressiveness: Агрессивность VAD (0-3)
        """
        self.transcriber = transcriber
        self.keyword = keyword.lower()
        self.sample_rate = sample_rate
        self.min_audio_duration = min_audio_duration
        self.energy_threshold = energy_threshold
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self._last_detection_time = 0.0
        self._cooldown_seconds = 2.0  # Защита от повторных срабатываний

    def is_wake_word_detected(self, audio: np.ndarray) -> bool:
        """Проверяет, обнаружен ли wake word в аудио.

        Args:
            audio: Массив аудио-данных

        Returns:
            True, если wake word обнаружен
        """
        import time

        # Проверка минимальной длительности
        duration = len(audio) / self.sample_rate
        if duration < self.min_audio_duration:
            return False

        # Предварительная фильтрация по энергии
        energy = np.mean(audio**2)
        if energy < self.energy_threshold:
            return False

        # Проверка VAD
        audio_int16 = (audio * 32767).astype(np.int16)
        if len(audio_int16) < int(self.sample_rate * 0.01):  # Минимум 10ms для VAD
            return False

        is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
        if not is_speech:
            return False

        # Защита от повторных срабатываний
        current_time = time.time()
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return False

        # Распознавание речи
        try:
            text = self.transcriber.transcribe(audio)
            text_lower = text.lower().strip()

            # Проверка наличия ключевого слова
            if self.keyword in text_lower:
                self._last_detection_time = current_time
                return True
        except Exception:
            # В случае ошибки распознавания возвращаем False
            pass

        return False


class WakeWordListener:
    """Слушатель wake word в реальном времени."""

    def __init__(
        self,
        detector: WakeWordDetectorProtocol,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
    ) -> None:
        """Инициализирует слушатель wake word.

        Args:
            detector: Детектор wake word
            sample_rate: Частота дискретизации
            channels: Количество каналов
            frame_duration_ms: Длительность фрейма в миллисекундах
        """
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self._callback: Callable[[], None] | None = None

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Устанавливает callback для обработки обнаруженного wake word.

        Args:
            callback: Функция, вызываемая при обнаружении wake word
        """
        self._callback = callback

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: sd.CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:
        """Внутренний callback для обработки аудио."""
        audio = indata[:, 0]
        if self.detector.is_wake_word_detected(audio) and self._callback:
            self._callback()

    def listen(self) -> None:
        """Начинает прослушивание wake word."""
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * self.frame_duration_ms / 1000),
            callback=self._audio_callback,
        ):
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                pass


class BufferedWakeWordListener:
    """Слушатель wake word с буферизацией аудио для распознавания ключевых слов."""

    def __init__(
        self,
        detector: WakeWordDetectorProtocol,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
        buffer_duration_seconds: float = 1.5,
    ) -> None:
        """Инициализирует слушатель wake word с буферизацией.

        Args:
            detector: Детектор wake word
            sample_rate: Частота дискретизации
            channels: Количество каналов
            frame_duration_ms: Длительность фрейма в миллисекундах
            buffer_duration_seconds: Длительность буфера в секундах
        """
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.buffer_duration_samples = int(sample_rate * buffer_duration_seconds)
        self._audio_buffer: list[np.ndarray] = []
        self._callback: Callable[[], None] | None = None

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Устанавливает callback для обработки обнаруженного wake word.

        Args:
            callback: Функция, вызываемая при обнаружении wake word
        """
        self._callback = callback

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: sd.CallbackTime,
        status: sd.CallbackFlags,
    ) -> None:
        """Внутренний callback для обработки аудио."""
        audio = indata[:, 0]
        self._audio_buffer.append(audio.copy())

        # Ограничиваем размер буфера
        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        while total_samples > self.buffer_duration_samples and len(self._audio_buffer) > 1:
            self._audio_buffer.pop(0)
            total_samples = sum(len(chunk) for chunk in self._audio_buffer)

        # Проверяем накопленное аудио
        if len(self._audio_buffer) > 0:
            buffered_audio = np.concatenate(self._audio_buffer)
            if self.detector.is_wake_word_detected(buffered_audio) and self._callback:
                self._callback()
                # Очищаем буфер после обнаружения
                self._audio_buffer.clear()

    def listen(self) -> None:
        """Начинает прослушивание wake word."""
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * self.frame_duration_ms / 1000),
            callback=self._audio_callback,
        ):
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                pass
