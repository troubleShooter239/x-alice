from vosk import KaldiRecognizer, Model


class VoskSpeechToText:
    __slots__ = ("_model", "_recognizer")

    def __init__(self, model_path: str, sample_rate: int) -> None:
        self._model = Model(model_path)
        self._recognizer = KaldiRecognizer(self._model, sample_rate)

    def transcribe(self, audio: bytes) -> str | None:
        return self._recognizer.Result()[14:-3] if self._recognizer.AcceptWaveform(audio) else None
