from src import config
from src.audio.recorder import AudioRecorder
from src.audio.stt import VoskSpeechToText
from src.audio.tts import SileroTextToSpeech
from src.router import CommandRouter
from src.wakeword.detector import FuzzyWakeWordDetector


class VoiceAssistant:
    __slots__ = ("stt", "tts", "router", "detector", "recorder")

    def __init__(
        self,
        stt: VoskSpeechToText,
        tts: SileroTextToSpeech,
        router: CommandRouter,
        detector: FuzzyWakeWordDetector,
        recorder: AudioRecorder,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.router = router
        self.detector = detector
        self.recorder = recorder

    def run(self) -> None:
        with self.recorder as mic:
            for chunk in mic:
                if not (text := self.stt.transcribe(chunk)):
                    continue

                print(f"Text: {text}")
                words = text.split()
                if self.detector.is_wake_word(words):
                    print("Wake word detected!")
                    self.router.route(text)
                else:
                    print("No wake word detected!")


def main() -> None:
    VoiceAssistant(
        stt=VoskSpeechToText(model_path="models/vosk-model-small-ru-0.22", sample_rate=config.SAMPLE_RATE),
        tts=SileroTextToSpeech(),
        router=CommandRouter(),
        detector=FuzzyWakeWordDetector(config.WAKE_KEYWORD, 0.75),
        recorder=AudioRecorder(config.SAMPLE_RATE, config.BLOCK_SIZE, channels=config.CHANNELS),
    ).run()


if __name__ == "__main__":
    main()
