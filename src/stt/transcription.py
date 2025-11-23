from typing import Literal, Protocol

import numpy as np
from faster_whisper import WhisperModel


class SpeechTranscriberProtocol(Protocol):
    def run(self, audio: np.ndarray, language: str | None, beam_size: int) -> str: ...


class SpeechTranscriber:
    def __init__(
        self,
        model_name: Literal[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "distil-small.en",
            "medium",
            "medium.en",
            "distil-medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-large-v3",
            "large-v3-turbo",
            "turbo",
        ] = "tiny",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        compute_type: Literal["default", "int8", "int16", "float16", "float32"] = "int8",
        language: str | None = None,
        beam_size: int = 1,
    ) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def run(self, audio: np.ndarray, language: str | None = None, beam_size: int = 1) -> str:
        segments, _ = self.model.transcribe(audio, beam_size=beam_size, language=language)
        return "".join([seg.text for seg in segments]).strip()
