from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np
import sounddevice as sd
import webrtcvad

from src.utils.types import Callback


if TYPE_CHECKING:
    from stt.transcription import SpeechTranscriberProtocol


class WakeWordDetectorProtocol(Protocol):
    def is_wake_word_detected(self, audio: np.ndarray) -> bool:
        pass


class WakeWordListenerProtocol(Protocol):
    """Protocol for wake word listener."""

    def set_callback(self, callback: Callback) -> None:
        pass

    def listen(self) -> None:
        pass


class KeywordWakeWordDetector:
    def __init__(
        self,
        transcriber: SpeechTranscriberProtocol,
        keyword: str = "алиса",
        sample_rate: int = 16000,
        min_audio_duration: float = 0.5,
        energy_threshold: float = 0.01,
        vad_aggressiveness: int = 2,
    ) -> None:
        """Initializes wake word detector based on keyword recognition.

        Args:
            transcriber: Speech transcriber for recognition
            keyword: Keyword for detection (default: "алиса")
            sample_rate: Sample rate
            min_audio_duration: Minimum audio duration for recognition (seconds)
            energy_threshold: Energy threshold for preliminary filtering
            vad_aggressiveness: VAD aggressiveness (0-3)
        """
        self.transcriber = transcriber
        self.keyword = keyword.lower()
        self.sample_rate = sample_rate
        self.min_audio_duration = min_audio_duration
        self.energy_threshold = energy_threshold
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self._last_detection_time = 0.0
        self._cooldown_seconds = 2.0

    def is_wake_word_detected(self, audio: np.ndarray) -> bool:
        if len(audio) / self.sample_rate < self.min_audio_duration:
            return False

        if np.mean(audio**2) < self.energy_threshold:
            return False

        audio_int16 = (audio * 32767).astype(np.int16)
        if len(audio_int16) < int(self.sample_rate * 0.01):
            return False

        if not self.vad.is_speech(audio_int16.tobytes(), self.sample_rate):
            return False

        current_time = time.time()
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return False

        try:
            if self.keyword in self.transcriber.transcribe(audio):
                self._last_detection_time = current_time
                return True
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        return False


class WakeWordListener:
    def __init__(
        self,
        detector: WakeWordDetectorProtocol,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
    ) -> None:
        """Initializes wake word listener.

        Args:
            detector: Wake word detector
            sample_rate: Sample rate
            channels: Number of channels
            frame_duration_ms: Frame duration in milliseconds
        """
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self._callback: Callback | None = None

    def set_callback(self, callback: Callback) -> None:
        self._callback = callback

    def _audio_callback(self, indata: np.ndarray, *args: Sequence[object]) -> None:
        audio = indata[:, 0]
        if self.detector.is_wake_word_detected(audio) and self._callback:
            self._callback()

    def listen(self) -> None:
        """Starts listening for wake word."""
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
    def __init__(
        self,
        detector: WakeWordDetectorProtocol,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
        buffer_duration_seconds: float = 1.5,
    ) -> None:
        """Initializes buffered wake word listener.

        Args:
            detector: Wake word detector
            sample_rate: Sample rate
            channels: Number of channels
            frame_duration_ms: Frame duration in milliseconds
            buffer_duration_seconds: Buffer duration in seconds
        """
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self.buffer_duration_samples = int(sample_rate * buffer_duration_seconds)
        self._audio_buffer: list[np.ndarray] = []
        self._callback: Callback | None = None

    def set_callback(self, callback: Callback) -> None:
        self._callback = callback

    def _audio_callback(self, indata: np.ndarray, *args: Sequence[object]) -> None:
        audio = indata[:, 0]
        self._audio_buffer.append(audio.copy())

        total_samples = sum(len(chunk) for chunk in self._audio_buffer)
        while total_samples > self.buffer_duration_samples and len(self._audio_buffer) > 1:
            self._audio_buffer.pop(0)
            total_samples = sum(len(chunk) for chunk in self._audio_buffer)

        if len(self._audio_buffer) > 0:
            buffered_audio = np.concatenate(self._audio_buffer)
            if self.detector.is_wake_word_detected(buffered_audio) and self._callback:
                self._callback()
                self._audio_buffer.clear()

    def listen(self) -> None:
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
