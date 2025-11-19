from src.assistant import VoiceAssistant
from src.recorder.audio import AudioRecorder
from src.stt.transcription import SpeechTranscriber
from src.wakeword.detection import BufferedWakeWordListener, KeywordWakeWordDetector


SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
LISTEN_DURATION = 3.0
WAKE_KEYWORD = "алиса"


def create_assistant() -> VoiceAssistant:
    audio_recorder = AudioRecorder(sample_rate=SAMPLE_RATE, channels=CHANNELS)
    transcriber = SpeechTranscriber(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="ru",
        beam_size=1,
    )
    wake_word_detector = KeywordWakeWordDetector(
        transcriber=transcriber,
        keyword=WAKE_KEYWORD,
        sample_rate=SAMPLE_RATE,
        min_audio_duration=0.5,
        energy_threshold=0.01,
        vad_aggressiveness=2,
    )
    wake_word_listener = BufferedWakeWordListener(
        detector=wake_word_detector,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        frame_duration_ms=FRAME_DURATION_MS,
        buffer_duration_seconds=1.5,
    )
    assistant = VoiceAssistant(
        audio_recorder=audio_recorder,
        transcriber=transcriber,
        wake_word_listener=wake_word_listener,
        listen_duration=LISTEN_DURATION,
    )
    return assistant


def main() -> None:
    assistant = create_assistant()
    assistant.start()


if __name__ == "__main__":
    main()
