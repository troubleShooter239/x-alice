from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from src.recorder.audio import AudioRecorderProtocol
    from src.stt.transcription import SpeechTranscriberProtocol
    from src.wakeword.detection import WakeWordListenerProtocol


class VoiceAssistantProtocol(Protocol):
    def start(self) -> None: ...


class VoiceAssistant:
    def __init__(
        self,
        audio_recorder: AudioRecorderProtocol,
        transcriber: SpeechTranscriberProtocol,
        wake_word_listener: WakeWordListenerProtocol,
        listen_duration: float = 3.0,
    ) -> None:
        self.audio_recorder = audio_recorder
        self.transcriber = transcriber
        self.wake_word_listener = wake_word_listener
        self.listen_duration = listen_duration

    def _on_wake_word_detected(self) -> None:
        print("🚀 Wakeword detected!")
        command_audio = self.audio_recorder.run(self.listen_duration)
        command_text = self.transcriber.run(command_audio, language="ru", beam_size=1)
        print("📝 Recognized:", command_text)

    def start(self) -> None:
        print("🎧 Listening for wake word... Say something loudly to activate.")
        self.wake_word_listener.set_callback(self._on_wake_word_detected)
        try:
            self.wake_word_listener.run()
        except KeyboardInterrupt:
            print("Exiting...")
