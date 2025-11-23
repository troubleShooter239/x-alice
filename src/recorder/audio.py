from typing import Protocol

import numpy as np
import sounddevice as sd


class RecorderProtocol(Protocol):
    def record(self, duration: float) -> np.ndarray: ...


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration: float) -> np.ndarray:
        print("🎙 Speak command...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()
        return audio.squeeze()
