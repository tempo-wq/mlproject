"""
Модуль транскрибации голосовых сообщений.
Использует faster-whisper — оптимизированную реализацию Whisper от OpenAI.

faster-whisper — это НЕ нейросетевая модель, которую мы обучаем.
Это готовый инструмент speech-to-text, используемый как внешняя утилита,
аналогично тому как используют API или библиотеку.
"""

import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Обёртка над faster-whisper для транскрибации аудио.
    
    Параметры:
        model_size: Размер модели Whisper ("tiny", "base", "small", "medium", "large-v2")
                    Рекомендуется "small" — хороший баланс скорость/качество
        device: "cpu" или "cuda"
        compute_type: "int8" (быстро/мало RAM), "float16", "float32"
    """

    SUPPORTED_SIZES = ("tiny", "base", "small", "medium", "large-v2", "large-v3")

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        if model_size not in self.SUPPORTED_SIZES:
            raise ValueError(f"model_size должен быть одним из: {self.SUPPORTED_SIZES}")

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

        logger.info(f"Transcriber инициализирован: model={model_size}, device={device}")

    def _load_model(self):
        """Ленивая загрузка модели при первом использовании."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(f"Загружаю модель Whisper '{self.model_size}'...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Модель загружена.")
        return self._model

    def transcribe(self, audio_path: str) -> Tuple[str, str, float]:
        """
        Транскрибирует аудиофайл в текст.

        Args:
            audio_path: Путь к аудиофайлу (ogg, mp3, wav, m4a и др.)

        Returns:
            Tuple[str, str, float]:
                - transcript: Расшифрованный текст
                - language: Определённый язык ("ru", "en", и т.д.)
                - duration: Длительность аудио в секундах
        """
        model = self._load_model()
        start_time = time.time()

        try:
            segments, info = model.transcribe(
                audio_path,
                beam_size=5,              # Beam search — выше = точнее, медленнее
                best_of=5,                # Кандидатов при сэмплировании
                vad_filter=True,          # Voice Activity Detection — игнорируем тишину
                vad_parameters={
                    "min_silence_duration_ms": 500,   # Минимальная пауза для сегментации
                    "speech_pad_ms": 200,             # Паддинг вокруг речи
                },
                condition_on_previous_text=True,      # Учитываем предыдущий контекст
                word_timestamps=False,
            )

            # Собираем все сегменты в единый текст
            text_parts = []
            duration = 0.0
            for segment in segments:
                text_parts.append(segment.text.strip())
                duration = max(duration, segment.end)

            transcript = " ".join(text_parts).strip()
            language = info.language
            elapsed = time.time() - start_time

            logger.info(
                f"Транскрибация завершена: {len(transcript.split())} слов, "
                f"язык={language}, аудио={duration:.1f}с, время={elapsed:.1f}с"
            )

            return transcript, language, duration

        except Exception as e:
            logger.error(f"Ошибка транскрибации: {e}", exc_info=True)
            raise

    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        """
        Расширенная транскрибация с временными метками сегментов.
        Полезно для отладки и экспериментов.
        """
        model = self._load_model()

        segments_data = []
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )

        for segment in segments:
            segments_data.append(
                {
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": segment.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "prob": round(w.probability, 3)}
                        for w in (segment.words or [])
                    ],
                }
            )

        return {
            "segments": segments_data,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "full_text": " ".join(s["text"] for s in segments_data),
        }