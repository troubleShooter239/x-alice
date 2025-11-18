"""Модуль голосового ассистента."""

from stt.transcription import SpeechTranscriberProtocol
from utils.audio import AudioRecorderProtocol
from wakeword.detection import WakeWordListenerProtocol


class VoiceAssistant:
    """Голосовой ассистент, оркестрирующий компоненты."""

    def __init__(
        self,
        audio_recorder: AudioRecorderProtocol,
        transcriber: SpeechTranscriberProtocol,
        wake_word_listener: WakeWordListenerProtocol,
        listen_duration: float = 3.0,
    ) -> None:
        """Инициализирует голосовой ассистент.

        Args:
            audio_recorder: Рекордер аудио
            transcriber: Транскрибер речи
            wake_word_listener: Слушатель wake word
            listen_duration: Длительность прослушивания команды в секундах
        """
        self.audio_recorder = audio_recorder
        self.transcriber = transcriber
        self.wake_word_listener = wake_word_listener
        self.listen_duration = listen_duration

    def _on_wake_word_detected(self) -> None:
        """Обработчик обнаружения wake word."""
        print("🚀 Wakeword detected!")
        command_audio = self.audio_recorder.record(self.listen_duration)
        command_text = self.transcriber.transcribe(command_audio)
        print("📝 Распознано:", command_text)

    def start(self) -> None:
        """Запускает ассистента."""
        print("🎧 Слушаем wakeword... Говори что-то громко, чтобы активировать.")
        self.wake_word_listener.set_callback(self._on_wake_word_detected)
        try:
            self.wake_word_listener.listen()
        except KeyboardInterrupt:
            print("Выход...")
