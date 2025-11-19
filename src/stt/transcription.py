from typing import Protocol

import numpy as np
from faster_whisper import WhisperModel


class SpeechTranscriberProtocol(Protocol):
    def transcribe(self, audio: np.ndarray) -> str:
        pass


class SpeechTranscriber:
    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 1,
    ) -> None:
        """Initializes transcriber.

        Args:
            model_name: Whisper model name
            device: Computing device (cpu/cuda)
            compute_type: Compute type
            language: Recognition language
            beam_size: Beam size for decoding
        """
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=self.beam_size, language=self.language)
        return "".join([seg.text for seg in segments]).strip()
