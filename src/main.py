from .assistant import VoiceAssistant
from .stt.transcription import SpeechTranscriber
from .utils.audio import AudioRecorder
from .wakeword.detection import BufferedWakeWordListener, KeywordWakeWordDetector


# Конфигурация
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
LISTEN_DURATION = 3.0
WAKE_KEYWORD = "алиса"


def create_assistant() -> VoiceAssistant:
    """Создает и настраивает голосовой ассистент.

    Returns:
        Настроенный экземпляр VoiceAssistant
    """
    audio_recorder = AudioRecorder(sample_rate=SAMPLE_RATE, channels=CHANNELS)
    transcriber = SpeechTranscriber(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="ru",
        beam_size=1,
    )
    # Создаем детектор wake word на основе распознавания ключевого слова
    wake_word_detector = KeywordWakeWordDetector(
        transcriber=transcriber,
        keyword=WAKE_KEYWORD,
        sample_rate=SAMPLE_RATE,
        min_audio_duration=0.5,
        energy_threshold=0.01,
        vad_aggressiveness=2,
    )
    # Используем буферизованный слушатель для накопления аудио
    wake_word_listener = BufferedWakeWordListener(
        detector=wake_word_detector,
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        frame_duration_ms=FRAME_DURATION_MS,
        buffer_duration_seconds=1.5,
    )

    # Создание ассистента с внедренными зависимостями
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
