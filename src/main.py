from src.assistant import VoiceAssistant
from src.recorder.audio import AudioRecorder
from src.stt.transcription import SpeechTranscriber
from src.wakeword.detection import WakeWordDetector, WakeWordListener


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
LISTEN_DURATION = 3.0
WAKE_KEYWORD = "алиса"


def main() -> None:
    audio_recorder = AudioRecorder(sample_rate=SAMPLE_RATE, channels=CHANNELS)
    transcriber = SpeechTranscriber(
        model_name="tiny",
        device="cpu",
        language="ru",
    )
    wake_word_detector = WakeWordDetector(
        transcriber=transcriber,
        keyword=WAKE_KEYWORD,
        sample_rate=SAMPLE_RATE,
    )
    wake_word_listener = WakeWordListener(
        detector=wake_word_detector,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        frame_duration_ms=FRAME_DURATION_MS,
    )
    assistant = VoiceAssistant(
        audio_recorder=audio_recorder,
        transcriber=transcriber,
        wake_word_listener=wake_word_listener,
        listen_duration=LISTEN_DURATION,
    )

    assistant.start()


if __name__ == "__main__":
    main()
