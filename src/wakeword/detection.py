from __future__ import annotations

from time import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
import sounddevice as sd
from rapidfuzz import fuzz
from webrtcvad import Vad


if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.utils.types import Callback
    from stt.transcription import TranscriberProtocol


class WakeWordDetectorProtocol(Protocol):
    def is_detected(self, audio: np.ndarray) -> bool: ...


class WakeWordListenerProtocol(Protocol):
    def set_callback(self, callback: Callback) -> None: ...
    def listen(self) -> None: ...


class WakeWordDetector:
    def __init__(
        self,
        transcriber: TranscriberProtocol,
        keyword: str,
        sample_rate: int = 16000,
        min_audio_duration: float = 0.5,
        energy_threshold: float = 0.01,
        vad_aggressiveness: int = 2,
        similarity_threshold: float = 70.0,
    ) -> None:
        self.transcriber = transcriber
        self.keyword = keyword.lower()
        self.sample_rate = sample_rate
        self.min_audio_duration = min_audio_duration
        self.energy_threshold = energy_threshold
        self.vad = Vad(vad_aggressiveness)
        self.similarity_threshold = similarity_threshold
        self._last_detection_time = 0.0
        self._cooldown_seconds = 2.0

    def is_detected(self, audio: np.ndarray) -> bool:
        if len(audio) / self.sample_rate < self.min_audio_duration:
            return False

        if np.mean(audio**2) < self.energy_threshold:
            return False

        audio_int16 = (audio * 32767).astype(np.int16)
        if len(audio_int16) < int(self.sample_rate * 0.01):
            return False

        if not self.vad.is_speech(audio_int16.tobytes(), self.sample_rate):
            return False

        current_time = time()
        if current_time - self._last_detection_time < self._cooldown_seconds:
            return False

        transcribed_text = self.transcriber.transcribe(audio)
        for word in transcribed_text.split():
            if fuzz.ratio(self.keyword, word) > self.similarity_threshold:
                self._last_detection_time = current_time
                return True

        return False


class WakeWordListener:
    def __init__(
        self,
        detector: WakeWordDetectorProtocol,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_duration_ms: int = 30,
    ) -> None:
        self.detector = detector
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        self._callback: Callback | None = None
        self._blocksize = int(self.sample_rate * self.frame_duration_ms / 1000)

    def set_callback(self, callback: Callback) -> None:
        self._callback = callback

    def _audio_callback(self, indata: np.ndarray, *_: Sequence[object]) -> None:
        audio = indata[:, 0]
        if self.detector.is_detected(audio) and self._callback:
            self._callback()

    def listen(self) -> None:
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self._blocksize,
            callback=self._audio_callback,
        ):
            try:
                while True:
                    sd.sleep(1000)
            except KeyboardInterrupt:
                pass
