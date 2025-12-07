from src import config
from src.audio.recorder import AudioRecorder
from src.audio.stt import VoskSpeechToText
from src.audio.tts import SileroTextToSpeech
from src.commands.handler import CommandHandler
from src.commands.text_filter import TextFilter
from src.wakeword.detector import FuzzyWakeWordDetector


class VoiceAssistant:
    __slots__ = ("stt", "tts", "handler", "detector", "recorder", "text_filter")

    def __init__(
        self,
        stt: VoskSpeechToText,
        tts: SileroTextToSpeech,
        handler: CommandHandler,
        detector: FuzzyWakeWordDetector,
        recorder: AudioRecorder,
        text_filter: TextFilter,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.handler = handler
        self.detector = detector
        self.recorder = recorder
        self.text_filter = text_filter

    def run(self) -> None:
        with self.recorder as mic:
            for chunk in mic:
                if not (text := self.stt.transcribe(chunk)):
                    continue

                print(f"Text: {text}")

                text = self.text_filter.filter_tbr(text)

                if self.detector.is_wake_word(text.split()):
                    print("Wake word detected!")
                    # TODO: handle commands
                else:
                    print("No wake word detected!")


def main() -> None:
    VoiceAssistant(
        stt=VoskSpeechToText(model_path="models/vosk-model-small-ru-0.22", sample_rate=config.SAMPLE_RATE),
        tts=SileroTextToSpeech(),
        handler=CommandHandler(),
        detector=FuzzyWakeWordDetector(config.WAKE_KEYWORD, 0.75),
        recorder=AudioRecorder(config.SAMPLE_RATE, config.BLOCK_SIZE, channels=config.CHANNELS),
        text_filter=TextFilter(config.VA_TBR),
    ).run()


if __name__ == "__main__":
    main()
